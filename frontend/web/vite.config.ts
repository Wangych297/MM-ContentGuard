/**
 * 文件功能：Vite 构建配置。启用 React 插件，配置开发服务器监听
 * 127.0.0.1:5173（禁用 HMR），将 /api 和 /media 请求代理到
 * 网关服务 http://127.0.0.1:8080。
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    hmr: false,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true
      },
      "/media": {
        target: "http://127.0.0.1:8080",
        changeOrigin: true
      }
    }
  }
});
