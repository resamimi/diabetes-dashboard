import React from 'react';

const FavoriteMessages = ({ favorites, onMessageClick }) => {
  return (
    <div className="favorite-messages">
      <h2>Favorite Messages</h2>
      <div className="favorite-messages-content">
        {favorites.map((fav) => (
          <div 
            key={fav.id} 
            className="favorite-message"
            onClick={() => onMessageClick(fav.id)}
            title="Click to scroll to message"
          >
            <div className="favorite-message-text">{fav.shortText}</div>
            <div className="favorite-message-time">{fav.timestamp}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FavoriteMessages;