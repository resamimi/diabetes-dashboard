

export const categories = [
  {
    name: 'Global Explanations',
    questions: [
      'What are the most important features for the predictions?',
      // 'Give me the reasons behind the prediction of samples with diabetes',
    ],
    important: true
  },
  {
    name: 'Individual Explanations',
    questions: [
      'Can you explain why the prediction of sample 39 was made?',
      'Why does my model predict those with glucose not less than 130 as true?',
      'What does instance with id 67 need to do to change the prediction?',
    ],
    // important: true,
    description: 'Get clear explanations for why specific results were given, based on patient measurements and medical data.<br><br><strong>Tip:</strong> You can easily find any sample to have an explanation for; just go to the Data Visualization tab and visualize the whole data.'
  },
  {
    name: 'Potential Errors',
    questions: [
      'what are some data points you get incorrect?',
      'what errors does the model typically make on the data?'
    ]
  },
  {
    name: 'Data Visualization',
    questions: [
      'Can you visualize the whole data?',
      'display instance with id 39?',
      // 'show me some instances where glucose is greater than 100'
    ],
    // important: true,
    description: 'See patient information displayed in clear charts and graphs. You can view all data at once or focus on specific patients and their test results.'
  },
  
  // {
  //     name: 'Recommendation Analysis',
  //     questions: [
  //       'What does instance with id 67 need to do to change the prediction?',
  //     ]
  // },
  {
    name: 'Assessment Results',
    questions: [
      'What is the model\'s prediction for the sample 55?',
      'Could you show me the predictions on all the data?',
      'What are the predictions for instances with glucose less than 50?'
    ]
  },
  {
    name: 'Result Probability',
    questions: [
      'What\'s the likelihood of being predicted as likely to have diabetes',
      'How likely are instances with glucose less than 80 predicted as likely to have diabetes?'
    ]
  },
  {
    name: 'System Capabilities',
    questions: [
      'What can you do?',
      'describe the model?',
    ]
  },
  {
    name: 'Accuracy Information',
    questions: [
      'give me the accuracy on the data',
      // 'can you show me the f1 score on the testing data?',
      'could you show model performance on glucose over 90?'
    ]
  },
  {
    name: 'What-If Analysis',
    questions: [
      'What the model would predict if you decreased glucose by 5 on all the data?',
      'What would the prediction for id 55 be if you change glucose to 80',
    ]
  },
  {
    name: 'Measurement Relationships',
    questions: [
      'What are the feature interactions for the model\'s predictions on the data?',
      'How do the features interact with each other on data with glucose equal to 100 or greater as true?'
    ]
  },
  {
    name: 'Previous Results',
    questions: [
      'what are the labels for all the data',
      'what are the ground truth labels in the data for glucose greater than 100',
    ]
  }
];