"""The main script that controls conversation logic.

This file contains the core logic for facilitating conversations. It orchestrates the necessary
routines for setting up conversations, controlling the state of the conversation, and running
the functions to get the responses to user inputs.
"""
import pickle
from random import seed as py_random_seed
import secrets
import numpy as np
import torch
from typing import List, Dict, Optional
from flask import Flask
import gin
import os
import json
import copy
from pprint import pprint
from explain.action import run_action
from explain.conversation import Conversation
from explain.decoder import Decoder
from explain.explanation import MegaExplainer, TabularDice
from explain.parser import Parser, get_parse_tree
from explain.prompts import Prompts
from explain.utils import read_and_format_data
from explain.write_to_log import log_dialogue_input
from explain.actions.filter import filter_operation
from explain.chat_history_manager import ChatHistoryManager
from explain.external_chatbot_answers import ClaudeAPIHandler, api_key
from prompt_redirection.semantic_matching import SemanticQueryMatcher


app = Flask(__name__)


@gin.configurable
def load_sklearn_model(filepath):
    """Loads a sklearn model."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model


@gin.configurable
class ExplainBot:
    """The ExplainBot Class."""

    def __init__(self,
                 model_file_path: str,
                 dataset_file_path: str,
                 background_dataset_file_path: str,
                 dataset_index_column: int,
                 target_variable_name: str,
                 categorical_features: List[str],
                 numerical_features: List[str],
                 remove_underscores: bool,
                 name: str,
                 parsing_model_name: str = "ucinlp/diabetes-t5-small",
                 seed: int = 0,
                 prompt_metric: str = "cosine",
                 prompt_ordering: str = "ascending",
                 t5_config: str = None,
                 use_guided_decoding: bool = True,
                 feature_definitions: dict = None,
                 skip_prompts: bool = False):
        """The init routine.

        Arguments:
            model_file_path: The filepath of the **user provided** model to explain. This model
                             should end with .pkl and support sklearn style functions like
                             .predict(...) and .predict_proba(...)
            dataset_file_path: The path to the dataset used in the conversation. Users will understand
                               the model's predictions on this dataset.
            background_dataset_file_path: The path to the dataset used for the 'background' data
                                          in the explanations.
            dataset_index_column: The index column in the data. This is used when calling
                                  pd.read_csv(..., index_col=dataset_index_column)
            target_variable_name: The name of the column in the dataset corresponding to the target,
                                  i.e., 'y'
            categorical_features: The names of the categorical features in the data. If None, they
                                  will be guessed.
            numerical_features: The names of the numeric features in the data. If None, they will
                                be guessed.
            remove_underscores: Whether to remove underscores in the feature names. This might help
                                performance a bit.
            name: The dataset name
            parsing_model_name: The name of the parsing model. See decoder.py for more details about
                                the allowed models.
            seed: The seed
            prompt_metric: The metric used to compute the nearest neighbor prompts. The supported options
                           are cosine, euclidean, and random
            prompt_ordering:
            t5_config: The path to the configuration file for t5 models, if using one of these.
            skip_prompts: Whether to skip prompt generation. This is mostly useful for running fine-tuned
                          models where generating prompts is not necessary.
        """

        # Set seeds
        np.random.seed(seed)
        py_random_seed(seed)
        torch.manual_seed(seed)

        # Initialize ChatHistoryManager without app
        self.history_manager = ChatHistoryManager()
        self.current_visualization = None
        self.current_patient = None

        self.bot_name = name

        # Prompt settings
        self.prompt_metric = prompt_metric
        self.prompt_ordering = prompt_ordering
        self.use_guided_decoding = use_guided_decoding

        # A variable used to help file uploads
        self.manual_var_filename = None

        self.decoding_model_name = parsing_model_name

        # Initialize completion + parsing modules
        app.logger.info(f"Loading parsing model {parsing_model_name}...")
        self.decoder = Decoder(parsing_model_name,
                               t5_config,
                               use_guided_decoding=self.use_guided_decoding,
                               dataset_name=name)

        # Initialize parser + prompts as None
        # These are done when the dataset is loaded
        self.prompts = None
        self.parser = None

        # Set up the conversation object
        self.conversation = Conversation(eval_file_path=dataset_file_path,
                                         feature_definitions=feature_definitions)

        # Load the model into the conversation
        self.load_model(model_file_path)

        # Load the dataset into the conversation
        self.load_dataset(dataset_file_path,
                          dataset_index_column,
                          target_variable_name,
                          categorical_features,
                          numerical_features,
                          remove_underscores,
                          store_to_conversation=True,
                          skip_prompts=skip_prompts)

        background_dataset = self.load_dataset(background_dataset_file_path,
                                               dataset_index_column,
                                               target_variable_name,
                                               categorical_features,
                                               numerical_features,
                                               remove_underscores,
                                               store_to_conversation=False)

        # Load the explanations
        self.load_explanations(background_dataset=background_dataset)

    def init_loaded_var(self, name: bytes):
        """Inits a var from manual load."""
        self.manual_var_filename = name.decode("utf-8")

    def load_explanations(self, background_dataset):
        """Loads the explanations.

        If set in gin, this routine will cache the explanations.

        Arguments:
            background_dataset: The background dataset to compute the explanations with.
        """
        app.logger.info("Loading explanations into conversation...")

        # This may need to change as we add different types of models
        pred_f = self.conversation.get_var('model_prob_predict').contents
        model = self.conversation.get_var('model').contents
        data = self.conversation.get_var('dataset').contents['X']
        categorical_f = self.conversation.get_var('dataset').contents['cat']
        numeric_f = self.conversation.get_var('dataset').contents['numeric']

        # Load lime tabular explanations
        mega_explainer = MegaExplainer(prediction_fn=pred_f,
                                       data=background_dataset,
                                       cat_features=categorical_f,
                                       class_names=self.conversation.class_names)
        mega_explainer.get_explanations(ids=list(data.index),
                                        data=data)
        message = (f"...loaded {len(mega_explainer.cache)} mega explainer "
                   "explanations from cache!")
        app.logger.info(message)
        # Load lime dice explanations
        tabular_dice = TabularDice(model=model,
                                   data=data,
                                   num_features=numeric_f,
                                   class_names=self.conversation.class_names)
        tabular_dice.get_explanations(ids=list(data.index),
                                      data=data)
        message = (f"...loaded {len(tabular_dice.cache)} dice tabular "
                   "explanations from cache!")
        app.logger.info(message)

        # Add all the explanations to the conversation
        self.conversation.add_var('mega_explainer', mega_explainer, 'explanation')
        self.conversation.add_var('tabular_dice', tabular_dice, 'explanation')

    def load_model(self, filepath: str):
        """Loads a model.

        This routine loads a model into the conversation
        from a specified file path. The model will be saved as a variable
        names 'model' in the conversation, overwriting an existing model.

        The routine determines the type of model from the file extension.
        Scikit learn models should be saved as .pkl's and torch as .pt.

        Arguments:
            filepath: the filepath of the model.
        Returns:
            success: whether the model was saved successfully.
        """
        app.logger.info(f"Loading inference model at path {filepath}...")
        if filepath.endswith('.pkl'):
            model = load_sklearn_model(filepath)
            self.conversation.add_var('model', model, 'model')
            self.conversation.add_var('model_prob_predict',
                                      model.predict_proba,
                                      'prediction_function')
        else:
            # No other types of models implemented yet
            message = (f"Models with file extension {filepath} are not supported."
                       " You must provide a model stored in a .pkl that can be loaded"
                       f" and called like an sklearn model.")
            raise NameError(message)
        app.logger.info("...done")
        return 'success'

    def load_dataset(self,
                     filepath: str,
                     index_col: int,
                     target_var_name: str,
                     cat_features: List[str],
                     num_features: List[str],
                     remove_underscores: bool,
                     store_to_conversation: bool,
                     skip_prompts: bool = False):
        """Loads a dataset, creating parser and prompts.

        This routine loads a dataset. From this dataset, the parser
        is created, using the feature names, feature values to create
        the grammar used by the parser. It also generates prompts for
        this particular dataset, to be used when determine outputs
        from the model.

        Arguments:
            filepath: The filepath of the dataset.
            index_col: The index column in the dataset
            target_var_name: The target column in the data, i.e., 'y' for instance
            cat_features: The categorical features in the data
            num_features: The numeric features in the data
            remove_underscores: Whether to remove underscores from feature names
            store_to_conversation: Whether to store the dataset to the conversation.
            skip_prompts: whether to skip prompt generation.
        Returns:
            success: Returns success if completed and store_to_conversation is set to true. Otherwise,
                     returns the dataset.
        """
        app.logger.info(f"Loading dataset at path {filepath}...")

        # Read the dataset and get categorical and numerical features
        dataset, y_values, categorical, numeric = read_and_format_data(filepath,
                                                                       index_col,
                                                                       target_var_name,
                                                                       cat_features,
                                                                       num_features)
                                                                    #    remove_underscores)


        if store_to_conversation:

            # Store the dataset
            self.conversation.add_dataset(dataset, y_values, categorical, numeric)

            # Set up the parser
            self.parser = Parser(cat_features=categorical,
                                 num_features=numeric,
                                 dataset=dataset,
                                 target=list(y_values))

            # Generate the available prompts
            # make sure to add the "incorrect" temporary feature
            # so we generate prompts for this
            self.prompts = Prompts(cat_features=categorical,
                                   num_features=numeric,
                                   target=np.unique(list(y_values)),
                                   feature_value_dict=self.parser.features,
                                   class_names=self.conversation.class_names,
                                   skip_creating_prompts=skip_prompts)
            app.logger.info("..done")

            return "success"
        else:
            return dataset

    def set_num_prompts(self, num_prompts):
        """Updates the number of prompts to a new number"""
        self.prompts.set_num_prompts(num_prompts)

    @staticmethod
    def gen_almost_surely_unique_id(n_bytes: int = 30):
        """To uniquely identify each input, we generate a random 30 byte hex string."""
        return secrets.token_hex(n_bytes)

    @staticmethod
    def log(logging_input: dict):
        """Performs the system logging."""
        assert isinstance(logging_input, dict), "Logging input must be dict"
        assert "time" not in logging_input, "Time field will be added to logging input"
        log_dialogue_input(logging_input)

    @staticmethod
    def build_logging_info(bot_name: str,
                           username: str,
                           response_id: str,
                           system_input: str,
                           parsed_text: str,
                           system_response: str):
        """Builds the logging dictionary."""
        return {
            'bot_name': bot_name,
            'username': username,
            'id': response_id,
            'system_input': system_input,
            'parsed_text': parsed_text,
            'system_response': system_response
        }

    def compute_parse_text(self, text: str, error_analysis: bool = False):
        """Computes the parsed text from the user text input.

        Arguments:
            error_analysis: Whether to do an error analysis step, where we compute if the
                            chosen prompts include all the
            text: The text the user provides to the system
        Returns:
            parse_tree: The parse tree from the formal grammar decoded from the user input.
            parse_text: The decoded text in the formal grammar decoded from the user input
                        (Note, this is just the tree in a string representation).
        """
        nn_prompts = None
        if error_analysis:
            grammar, prompted_text, nn_prompts = self.compute_grammar(text, error_analysis=error_analysis)
        else:
            grammar, prompted_text = self.compute_grammar(text, error_analysis=error_analysis)
        app.logger.info("About to decode")
        # Do guided-decoding to get the decoded text
        api_response = self.decoder.complete(
            prompted_text, grammar=grammar)
        decoded_text = api_response['generation']

        app.logger.info(f'Decoded text {decoded_text}')

        # Compute the parse tree from the decoded text
        # NOTE: currently, we're using only the decoded text and not the full
        # tree. If we need to support more complicated parses, we can change this.
        parse_tree, parsed_text = get_parse_tree(decoded_text)
        if error_analysis:
            return parse_tree, parsed_text, nn_prompts
        else:
            return parse_tree, parsed_text,

    def compute_parse_text_t5(self, text: str):
        """Computes the parsed text for the input using a t5 model.

        This supposes the user has finetuned a t5 model on their particular task and there isn't
        a need to do few shot
        """
        grammar, prompted_text = self.compute_grammar(text)
        decoded_text = self.decoder.complete(text, grammar)
        app.logger.info(f"t5 decoded text {decoded_text}")
        parse_tree, parse_text = get_parse_tree(decoded_text[0])
        return parse_tree, parse_text

    def compute_grammar(self, text, error_analysis: bool = False):
        """Computes the grammar from the text.

        Arguments:
            text: the input text
            error_analysis: whether to compute extra information used for error analyses
        Returns:
            grammar: the grammar generated for the input text
            prompted_text: the prompts computed for the input text
            nn_prompts: the knn prompts, without extra information that's added for the full
                        prompted_text provided to prompt based models.
        """
        nn_prompts = None
        app.logger.info("getting prompts")
        # Compute KNN prompts
        if error_analysis:
            prompted_text, adhoc, nn_prompts = self.prompts.get_prompts(text,
                                                                        self.prompt_metric,
                                                                        self.prompt_ordering,
                                                                        error_analysis=error_analysis)
        else:
            prompted_text, adhoc = self.prompts.get_prompts(text,
                                                            self.prompt_metric,
                                                            self.prompt_ordering,
                                                            error_analysis=error_analysis)
        app.logger.info("getting grammar")
        # Compute the formal grammar, making modifications for the current input
        grammar = self.parser.get_grammar(
            adhoc_grammar_updates=adhoc)

        if error_analysis:
            return grammar, prompted_text, nn_prompts
        else:
            return grammar, prompted_text

        
    def get_frontend_code(self, visualization_type: str, frontend_dir: str = "static/react/chat-interface/src/components") -> Optional[str]:
        """
        Gets frontend visualization component code based on visualization type.
        
        Args:
            visualization_type: Type of visualization (e.g., 'feature_importance')
            frontend_dir: Path to frontend components directory
        
        Returns:
            Component code if found, None otherwise
        """
        # Mapping of visualization types to component files
        component_files = {
            "feature_importance": "FeatureImportancePlot.js",
            "feature_range": "FeatureRangePlot.js", 
            "classification_scatter": "ScatterClassificationPlot.js",
            "typical_mistakes": "TypicalMistakesPlot.js",
            "counterfactual_explanation": "CounterfactualPlot.js",
            "individual_data": "IndividualDataPlot.js"
        }
        
        # Get component filename and construct path
        component_file = component_files.get(visualization_type)
        if not component_file:
            print(f"Warning: No component found for type '{visualization_type}'")
            return None
            
        file_path = os.path.join(frontend_dir, component_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Component file not found at '{file_path}'")
            return None
        except Exception as e:
            print(f"Error reading component file: {str(e)}")
            return None


    def ask_external_chatbot(self, current_query: str, user_session_conversation: Conversation):
        """
        Ask external chatbot (Claude) for response when local LLM cannot handle the query.
        Provides context including chat history, current patient data, and active visualization.
        
        Args:
            current_query: Current user question
            user_session_conversation: Current conversation session for the user
            
        Returns:
            str: Response from Claude
        """
        username = user_session_conversation.username
        
        # Get recent chat history for context
        recent_exchanges = self.history_manager.get_recent_exchanges(username, limit=3)
        
        complete_recent_exchanges = []
        for exchangeIdx, exchange in enumerate(recent_exchanges):
            visualization_data = recent_exchanges[exchangeIdx][2]
            if visualization_data == None:
                frontend_code = None
            else:
                frontend_code = self.get_frontend_code(visualization_data["type"])
            exchange.append(frontend_code)
            complete_recent_exchanges.append(exchange)

        # Get current patient data if available
        current_patient_data = None
        current_visualization_data = None
        current_visualization_code = None

        try:
            # self.get_patient_id(self.current_patient)

            # Check if there's active patient data in the conversation
            # if hasattr(user_session_conversation, 'temp_dataset') and user_session_conversation.temp_dataset:
            current_patient_data = {
                'patient_features': None,
                'model_prediction': None
            }
            
            # Get model prediction for current patient
            current_patient_data['model_prediction'] = self.get_prediction(self.current_patient)
            current_patient_data['patient_features'] = user_session_conversation.temp_dataset.contents['X'].to_dict()
            
            # Check for active visualization in the conversation
                
            # if self.current_visualization:
                # Get visualization data
            current_visualization_data = self.get_visualization_data(
                self.current_visualization, 
                str(user_session_conversation.temp_dataset.contents['X'].index[0])
            )
            
            # Get visualization component code
            current_visualization_code = self.get_frontend_code(self.current_visualization)
                    
        except Exception as e:
            app.logger.warning(f"Error getting current context data: {str(e)}")
            # Continue even if we can't get current context

        # Format context for Claude
        context = {
            "chat_history": complete_recent_exchanges,
            "current_patient": current_patient_data,
            "active_visualization": {
                "type": self.current_visualization,
                "data": current_visualization_data,
                "component_code": current_visualization_code
            }
        }

        # Initialize Claude handler
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
        handler = ClaudeAPIHandler(api_key)

        try:
            response = handler.get_response(
                current_query=current_query,
                previous_exchanges=complete_recent_exchanges,
                current_context=context
            )
        except Exception as e:
            app.logger.error(f"Error getting response from Claude: {str(e)}")
            raise

        return response


    def update_state(self, text: str, user_session_conversation: Conversation):
        """The main conversation driver."""
        try:
            app.logger.info("Starting update_state method")
            
            # Check initial conditions
            if text is None:
                app.logger.error("Text input is None")
                return ''
            if self.prompts is None:
                app.logger.error("Self.prompts is None")
                return ''
            if self.parser is None:
                app.logger.error("Self.parser is None")
                return ''

            app.logger.info(f'USER INPUT: {text}')

            # Load matcher
            app.logger.info("Loading semantic matcher")
            try:
                loaded_matcher = SemanticQueryMatcher.load("prompt_redirection/semantic_matcher")
                app.logger.info("Successfully loaded semantic matcher")
            except Exception as matcher_error:
                app.logger.error(f"Error loading semantic matcher: {str(matcher_error)}")
                raise

            # Test matcher
            app.logger.info("Testing matcher with input")
            try:
                is_supported, score, best_match = loaded_matcher.find_best_match(text)
                app.logger.info(f"Matcher results - Supported: {is_supported}, Score: {score:.4f}, Best match: {best_match}")
            except Exception as match_error:
                app.logger.error(f"Error in matcher.find_best_match: {str(match_error)}")
                raise

            if is_supported:
                app.logger.info("Query is supported by internal parser")
                try:
                    # Parse user input
                    if "t5" not in self.decoding_model_name:
                        app.logger.info("Using standard parse")
                        parse_tree, parsed_text = self.compute_parse_text(text)
                    else:
                        app.logger.info("Using t5 parse")
                        parse_tree, parsed_text = self.compute_parse_text_t5(text)
                    
                    app.logger.info(f"Successfully parsed text: {parsed_text}")

                    # Run action
                    app.logger.info("Running action with parsed text")
                    returned_item = run_action(user_session_conversation, parse_tree, parsed_text)
                    app.logger.info("Successfully ran action")
                    
                except Exception as parse_error:
                    app.logger.error(f"Error in parsing/action: {str(parse_error)}")
                    app.logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise

            else:
                app.logger.info("Query not supported, redirecting to external chatbot")
                try:
                    parsed_text = ""
                    returned_item = self.ask_external_chatbot(text, user_session_conversation)
                    app.logger.info("Successfully got response from external chatbot")
                except Exception as chatbot_error:
                    app.logger.error(f"Error in external chatbot: {str(chatbot_error)}")
                    app.logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise

            # Log response details
            app.logger.info("Preparing final response")
            try:
                username = user_session_conversation.username
                response_id = self.gen_almost_surely_unique_id()
                
                logging_info = self.build_logging_info(
                    self.bot_name,
                    username,
                    response_id,
                    text,
                    parsed_text,
                    returned_item
                )
                
                self.log(logging_info)
                app.logger.info("Successfully logged response")
                
                # Create final result
                final_result = returned_item + f"<>{response_id}"
                app.logger.info("Final result prepared")
                
                return final_result
                
            except Exception as final_error:
                app.logger.error(f"Error in final response preparation: {str(final_error)}")
                app.logger.error(f"Full traceback: {traceback.format_exc()}")
                raise
                
        except Exception as e:
            app.logger.error(f"Unhandled error in update_state: {str(e)}")
            app.logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"An error occurred while processing your request.<>{str(id('error'))}"

    def get_patient_id(self, patient_id_str):

        try:

            parse_text = f"filter id {patient_id_str} [e]"
            returned_item = run_action(self.conversation, None, parse_text)


            # filter_operation(self.conversation, parse_text, 0)
            # patient_id = int(patient_id_str)
            # patient_data = all_data.loc[patient_id]

        except KeyError:
            raise ValueError(f"Patient ID {patient_id} not found in dataset")
            app.logger.error(f"Patient ID {patient_id} not found in dataset")

        except Exception as e:
            raise Exception(f"Error retrieving patient data: {str(e)}")
            app.logger.error(f"Error retrieving patient data: {str(e)}")


    def get_patient_data(self, patient_id_str):
        """Get data for a specific patient including feature distributions."""
       
        self.get_patient_id(patient_id_str)
        patient_data = self.conversation.temp_dataset.contents['X']
        features = list(patient_data.columns)

        self.conversation.build_temp_dataset()
        all_data = self.conversation.temp_dataset.contents['X']

        # Format data including distributions
        formatted_data = {
            "data": {
                "sample_data": {},
                "max_values": {},
                "min_values": {},
                "distributions": {}
            }
        }
        
        # For each feature, add patient value and calculate distribution
        for feature in features:
            # Add patient value
            formatted_data["data"]["sample_data"][feature] = {
                patient_id_str: float(patient_data[feature].iloc[0])
            }
            
            # Add min/max values
            formatted_data["data"]["max_values"][feature] = float(all_data[feature].max())
            formatted_data["data"]["min_values"][feature] = float(all_data[feature].min())
            
            # Calculate distribution with 20 bins
            hist, bin_edges = np.histogram(all_data[feature], bins=20, density=False)
            formatted_data["data"]["distributions"][feature] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
            
        return formatted_data
            

    def get_prediction(self, patient_id_str):

        self.get_patient_id(patient_id_str)
        patient_data = self.conversation.temp_dataset.contents['X']
        model = self.conversation.get_var('model').contents

        # # Reshape the data to 2D array (1 sample × n_features)
        # patient_data = patient_data.to_numpy().reshape(1, -1)

        # Get class prediction
        pred = model.predict(patient_data)[0]

        # Get prediction probability 
        prob = model.predict_proba(patient_data)[0][int(pred)]

        
        return {
            'prediction': int(pred),
            'probability': round(float(prob), 1)  # Use highest probability
        }

    def get_visualization_data(self, viz_type, patient_id_str):
        """Get visualization data based on type and patient ID."""

        self.current_visualization = viz_type
        self.current_patient = patient_id_str

        try:
            if viz_type == "feature_importance":

                parse_text = f"filter id {patient_id_str} and explain features [e]"
                returned_item = run_action(self.conversation, None, parse_text)
                
            elif viz_type == "feature_range":

                parse_text = f"filter id {patient_id_str} and explain features range [e]"
                returned_item = run_action(self.conversation, None, parse_text)
                
            elif viz_type == "classification_scatter":
                
                parse_text = "data [e]"
                returned_item = run_action(self.conversation, None, parse_text)
                
            elif viz_type == "typical_mistakes":

                parse_text = "mistake typical [e]"
                returned_item = run_action(self.conversation, None, parse_text)
                
            elif viz_type == "counterfactual_explanation":

                parse_text = f"filter id {patient_id_str} and explain cfe [e]"
                returned_item = run_action(self.conversation, None, parse_text)

            
            start_idx = returned_item.find('<json>') + len('<json>')
            json_str = returned_item[start_idx:]

            # Parse the JSON string back into a Python object
            visualization_data = json.loads(json_str)

            return visualization_data
                            
        except Exception as e:
            app.logger.error(f"Error generating visualization data: {str(e)}")
            raise Exception(f"Error generating visualization data: {str(e)}")


    def get_what_if_prediction(self, patient_id_str, new_values):

        try:

            self.get_patient_id(patient_id_str)
            patient_data = self.conversation.temp_dataset.contents['X']
            model = self.conversation.get_var('model').contents

            # # Reshape the data to 2D array (1 sample × n_features)
            # patient_data = patient_data.to_numpy().reshape(1, -1)

            # Get the original patient data
            original_data = copy.deepcopy(patient_data)
            
            # Update the values for the specified patient
            for feature, value in new_values.items():
                original_data.at[int(patient_id_str), feature] = value

            # Get class prediction
            pred = model.predict(original_data)[0]

            # Get prediction probability 
            prob = model.predict_proba(original_data)[0][int(pred)]

            
            return {
                'prediction': int(pred),
                'probability': round(float(prob), 1)  # Use highest probability
            }

        except Exception as e:
            app.logger.error(f"Error generating visualization data: {str(e)}")
            raise Exception(f"Error generating visualization data: {str(e)}")


