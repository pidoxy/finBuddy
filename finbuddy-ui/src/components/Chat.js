import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './Chat.css';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const chatEndRef = useRef(null);
  const [isFirstRender, setIsFirstRender] = useState(true);

  useEffect(() => {
    // Scroll to the bottom of the chat when messages update
    if (!isFirstRender) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    } else {
        setIsFirstRender(false)
    }
  }, [messages]);

  const sendMessage = async () => {
      if (!input.trim()) return;
      
      const newMessage = { text: input, sender: 'user' };
      setMessages([...messages, newMessage]);
      setInput('');

      try {
          const response = await axios.post('http://localhost:5508/chat', {
          question: input,
          });

          const botMessage = { text: response.data.answer, sender: 'bot' };
          setMessages([...messages, newMessage, botMessage]);
      } catch (error) {
          console.error('Error sending message', error);
          const botMessage = { text: 'Oops! Something went wrong.', sender: 'bot' };
          setMessages([...messages, newMessage, botMessage]);
      }
  };

  return (
    <div className="chat-container">
      <div className="message-list">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {message.text}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => { if (e.key === 'Enter') sendMessage(); }}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
};

export default Chat;