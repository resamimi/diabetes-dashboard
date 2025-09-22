# Visual-Conversational Interface for Evidence-Based Explanation of Diabetes Risk Prediction

A clinical decision support system that combines interactive visualizations with conversational AI to explain diabetes risk assessments for healthcare professionals. This system integrates scientific evidence with AI explanations to support clinical decision-making.

## 🔬 About the Research

This system is based on the research paper "Visual-Conversational Interface for Evidence-Based Explanation of Diabetes Risk Prediction" presented at CUI '25. The system addresses key limitations in existing clinical decision support systems by providing:

- **Interactive Visualizations** with conversational explanations
- **Scientific Evidence Integration** to calibrate trust in AI decisions
- **Hybrid Query Processing** combining specialized models with general LLMs
- **Feature Range Analysis** comparing AI-observed ranges with medical guidelines

## 🏗️ System Architecture

The system consists of:
- **Frontend**: React-based dashboard with interactive visualizations
- **Backend**: Flask application with AI explanation capabilities
- **ML Model**: Gradient boosted tree for diabetes risk prediction
- **Conversational AI**: Integration with Claude API for natural language explanations
- **Scientific Evidence**: Pre-verified medical literature database

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18.x** and **npm 9.x**
- **Git** for cloning the repository

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/resamimi/diabetes-dashboard.git
   cd diabetes-dashboard
   ```

2. **Set Up Python Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Set Up Frontend**
   ```bash
   cd static/react/chat-interface
   npm install
   npm run build
   cd ../../..
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file in the root directory:
   ```bash
   # Required: Anthropic API key for conversational AI
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   
   # Optional: Azure Translator (for multi-language support)
   AZURE_TRANSLATOR_KEY=your_azure_translator_key
   AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
   AZURE_TRANSLATOR_REGION=your_region
   
   # Database configuration (optional - defaults to SQLite)
   DATABASE_URL=sqlite:///chat_history.db
   
   # Flask configuration
   FLASK_ENV=development
   PORT=7860
   ```

5. **Initialize the System**
   ```bash
   # Preload model and data
   python preload.py
   ```

### Running the Application

#### Option 1: Development Mode
```bash
# Run Flask development server
python flask_app.py
```

#### Option 2: Production Mode (using Gunicorn)
```bash
gunicorn --bind 0.0.0.0:7860 --timeout 300 --workers 1 --threads 4 flask_app:app
```

The application will be available at `http://localhost:7860`

### Using Docker (Alternative)

1. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**
   ```bash
   docker build -t diabetes-dashboard .
   docker run -p 7860:7860 diabetes-dashboard
   ```

## 📊 Data and Models

### Dataset
The system uses the **Pima Indians Diabetes Database** from the UCI Repository:
- 768 patient samples
- Features: Glucose, BMI, Blood Pressure, Age, etc.
- Binary classification: Diabetes risk (High/Low)

### Pre-trained Model
- **Algorithm**: Gradient Boosted Tree
- **Accuracy**: 73.3% on test set
- **Location**: `data/diabetes_model_grad_tree.pkl`

The model and dataset are automatically loaded during system initialization.

## 🎯 Key Features

### 1. Patient Data Visualization
- Interactive health factor displays
- Clinical threshold indicators
- Distribution context for each metric

### 2. Analysis Visualizations
- **Factor Importance**: Shows how each health factor influences AI assessment
- **Factor Range Analysis**: Compares AI-observed ranges with scientific ranges
- **Recommendations**: Step-by-step risk reduction guidance

### 3. Conversational AI Assistant
- Natural language explanations of AI decisions
- Scientific evidence integration with citations
- Follow-up questions and clarifications

### 4. Evidence-Based Explanations
- Scientific citations from medical literature
- Comparison between AI findings and clinical guidelines
- Hover-over citation details

## 💻 Usage Guide

### For Healthcare Professionals

1. **Patient Analysis**
   - Enter patient ID in the left panel
   - View risk assessment and health metrics
   - Examine factor importance and ranges

2. **Ask Questions**
   - Use the AI assistant to ask about specific factors
   - Request scientific evidence for explanations
   - Get personalized recommendations

3. **Explore Visualizations**
   - Click help buttons (🔵) next to factors for detailed explanations
   - Switch between different analysis types
   - Access scientific references through citations

### Sample Queries
- "Why is glucose more important than blood pressure for diabetes risk?"
- "What are the scientific guidelines for BMI in diabetes assessment?"
- "Can you provide specific dietary recommendations for this patient?"

## 🔧 Configuration

### Adding New Scientific Evidence
Evidence is stored in `cache/scientific_info.txt`. To update:
1. Modify the scientific information
2. Clear the cache file
3. Restart the system to regenerate evidence

### Customizing Visualizations
Visualization components are in `static/react/chat-interface/src/components/`:
- `FeatureImportancePlot.js` - Factor importance visualization
- `FeatureRangePlot.js` - Range analysis visualization  
- `CounterfactualTimeline.js` - Recommendations visualization
- `PatientDataPlot.js` - Patient data display

### Model Replacement
To use a different model:
1. Place your `.pkl` model file in `data/`
2. Update the model path in `global_config.gin`
3. Ensure your model supports `.predict()` and `.predict_proba()` methods

## 🌐 Multi-language Support

The system supports multiple languages through Azure Translator:
- English (default)
- Slovenian
- Additional languages can be configured in `TranslationProvider.js`

## 📋 API Endpoints

### Core Endpoints
- `GET /` - Main dashboard interface
- `POST /get_bot_response` - Conversational AI responses
- `GET /api/patient/<id>` - Patient data retrieval
- `GET /api/prediction/<id>` - Risk prediction
- `GET /api/visualization/<type>/<id>` - Visualization data

### Authentication Endpoints
- `POST /api/auth/signup` - User registration
- `POST /api/auth/signin` - User login
- `POST /api/auth/signout` - User logout

## 🔍 Troubleshooting

### Common Issues

1. **"Module not found" errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Frontend build failures**
   ```bash
   cd static/react/chat-interface
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

3. **API key errors**
   - Verify your `ANTHROPIC_API_KEY` in the `.env` file
   - Check that the API key has proper permissions

4. **Database connection issues**
   - For SQLite: Check write permissions in the project directory
   - For PostgreSQL: Verify `DATABASE_URL` format and connectivity

5. **Model loading errors**
   ```bash
   # Re-run preload script
   python preload.py
   ```

### Performance Optimization

- Use production WSGI server (Gunicorn) for better performance
- Enable caching for scientific evidence lookups
- Consider using Redis for session management in production

## 📚 Development

### Project Structure
```
diabetes-dashboard/
├── static/react/chat-interface/    # Frontend React application
├── explain/                        # Core explanation logic
├── data/                          # Models and datasets
├── cache/                         # Cached explanations and evidence
├── configs/                       # Configuration files
├── flask_app.py                   # Main Flask application
├── preload.py                     # System initialization
└── requirements.txt               # Python dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Testing
```bash
# Run backend tests (if available)
python -m pytest tests/

# Test frontend components
cd static/react/chat-interface
npm test
```

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.

## 🙏 Citation

If you use this system in your research, please cite:

```bibtex
@inproceedings{samimi2025visual,
  title={Visual-Conversational Interface for Evidence-Based Explanation of Diabetes Risk Prediction},
  author={Samimi, Reza and Bhattacharya, Aditya and Gosak, Lucija and Stiglic, Gregor and Verbert, Katrien},
  booktitle={Proceedings of the 7th ACM Conference on Conversational User Interfaces (CUI '25)},
  year={2025},
  organization={ACM}
}
```

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: resamimi@outlook.com

## 🔗 Links

- [Research Paper](https://doi.org/10.1145/3719160.3736616)
- [Demo Video](https://www.youtube.com/watch?v=dp8cU2V787w)