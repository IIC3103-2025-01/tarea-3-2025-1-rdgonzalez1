// frontend/src/Chat.js

import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./Chat.css";

function Chat({ docId }) {
  const [messages, setMessages] = useState([
    { from: "bot", text: "Article loaded successfully. What would you like to know?" }
  ]);
  const [input, setInput] = useState("");
  const [loadingQ, setLoadingQ] = useState(false);
  const [errorQ, setErrorQ] = useState("");
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorQ("");
    if (!input.trim()) {
      setErrorQ("Please type your question before sending.");
      return;
    }
    const question = input.trim();
    setMessages((prev) => [...prev, { from: "user", text: question }]);
    setInput("");
    setLoadingQ(true);

    try {
      const res = await axios.post("/query", { question });
      const answer = res.data.answer;
      setMessages((prev) => [...prev, { from: "bot", text: answer }]);
      setErrorQ("");
    } catch (err) {
      console.error(err.response || err);
      const detail = err.response?.data?.detail || "Unknown error querying the chatbot.";
      setErrorQ(detail);
      setMessages((prev) => [...prev, { from: "bot", text: "Sorry, an error occurred." }]);
    } finally {
      setLoadingQ(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={m.from === "user" ? "message user" : "message bot"}
          >
            {m.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {errorQ && <p className="error">{errorQ}</p>}

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question..."
          disabled={loadingQ}
        />
        <button type="submit" disabled={loadingQ}>
          {loadingQ ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
}

export default Chat;
