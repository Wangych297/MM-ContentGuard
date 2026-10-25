/**
 * 文件功能：前端应用入口。将 App 根组件挂载到 DOM 的 #root 元素，
 * 启用 React.StrictMode 严格模式。
 */
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
