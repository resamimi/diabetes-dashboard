

import React from 'react';

const WrongAnswerDialog = ({ isOpen, onClose, onSelect, messageId }) => {
  if (!isOpen) return null;

  const handleSelect = (reason) => {
    onSelect(messageId, `Wrong Answer - ${reason}`);
    onClose();
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h3>What went wrong?</h3>
        <div className="modal-buttons">
          <button onClick={() => handleSelect('Did not understand question')}>
            Chatbot didn't understand the question
          </button>
          <button onClick={() => handleSelect('Irrelevant answer')}>
            Answer is irrelevant
          </button>
        </div>
      </div>
    </div>
  );
};

export default WrongAnswerDialog;