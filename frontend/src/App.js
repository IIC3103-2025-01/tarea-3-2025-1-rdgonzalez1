// frontend/src/App.js

import React, { useState } from "react";
import axios from "axios";
import Chat from "./Chat";
import "./App.css";

function App() {
  const [url, setUrl] = useState("");
  const [loadingArticle, setLoading] = useState(false);
  const [articleLoaded, setArticleLoaded] = useState(false);
  const [docId, setDocId] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const handleUpload = async (e) => {
    e.preventDefault();
    setErrorMsg("");
    if (!url) {
      setErrorMsg("You must enter the Wikipedia article URL.");
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post("/upload-article", { url });
      setDocId(res.data.doc_id);
      setArticleLoaded(true);
      setErrorMsg("");
    } catch (err) {
      console.error(err.response || err);
      const detail = err.response?.data?.detail || "Unknown error loading the article.";
      setErrorMsg(detail);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    // Reset all state to allow new article upload
    setUrl("");
    setDocId("");
    setArticleLoaded(false);
    setErrorMsg("");
  };

  return (
    <div className="App">
      <h1>Wikipedia RAG Chatbot</h1>
      {!articleLoaded ? (
        <>
          <form onSubmit={handleUpload} className="url-form">
            <label htmlFor="urlInput">Enter Wikipedia URL (English):</label>
            <input
              id="urlInput"
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://en.wikipedia.org/wiki/..."
            />
            <button type="submit" disabled={loadingArticle}>
              {loadingArticle ? "Loading..." : "Load Article"}
            </button>
          </form>
          {errorMsg && <p className="error">{errorMsg}</p>}
        </>
      ) : (
        <>
          <div className="restart-container">
            <button onClick={handleReset} className="restart-button">
              Restart Chat
            </button>
          </div>
          <Chat docId={docId} />
        </>
      )}
    </div>
  );
}

export default App;
