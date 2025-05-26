
import * as React from "react";

const AlertDialog = ({ open, onOpenChange, children }) => {
  if (!open) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={() => onOpenChange?.(false)}
    >
      <div className="fixed inset-0 bg-black/50" /> {/* Backdrop */}
      <div 
        className="relative bg-white rounded-lg w-full max-w-2xl p-6 shadow-lg"
        onClick={e => e.stopPropagation()} // Prevent closing when clicking dialog content
      >
        {children}
      </div>
    </div>
  );
};

const AlertDialogContent = ({ children, className = "" }) => {
  return (
    <div className={`relative ${className}`}>
      {children}
    </div>
  );
};

const AlertDialogHeader = ({ children }) => {
  return (
    <div className="mb-4">
      {children}
    </div>
  );
};

const AlertDialogTitle = ({ children, className = "" }) => {
  return (
    <h2 className={`text-lg font-semibold ${className}`}>
      {children}
    </h2>
  );
};

const AlertDialogDescription = ({ children, className = "" }) => {
  return (
    <div className={`mt-2 text-sm text-gray-600 ${className}`}>
      {children}
    </div>
  );
};

const AlertDialogFooter = ({ children }) => {
  return (
    <div className="mt-4 flex justify-end space-x-2">
      {children}
    </div>
  );
};

const AlertDialogAction = ({ children, onClick, className = "" }) => {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${className}`}
    >
      {children}
    </button>
  );
};

export {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogAction,
};