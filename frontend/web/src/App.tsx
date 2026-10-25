/**
 * 文件功能：多模态聊天审核前端主组件。实现聊天界面，支持文本输入、
 * 图片/视频拖拽上传与文件选择，消息气泡展示，审核进度指示器，
 * 以及审核不通过原因的详情面板。
 *
 * 核心组件/函数：
 * - App: React 函数组件，包含完整的聊天交互和审核状态管理
 */
import {
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type RefObject
} from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Image as ImageIcon,
  Loader2,
  MessageCircle,
  MoreHorizontal,
  Phone,
  Play,
  Search,
  Send,
  Settings,
  Smile,
  User,
  Video,
  X
} from "lucide-react";

type ReviewReason = {
  modality: string;
  code: string;
  display: string;
  score?: number | null;
  source?: string | null;
  raw_reason?: string | null;
};

type ReviewDecision = {
  allow: boolean;
  status: string;
  summary: string;
  reasons: ReviewReason[];
};

type AttachmentRecord = {
  id: string;
  message_id: string;
  modality: "image" | "video" | string;
  original_name: string;
  mime_type: string;
  storage_path: string;
  size_bytes: number;
};

type MessageRecord = {
  id: string;
  client_message_id: string;
  conversation_id: string;
  sender_id: string;
  text?: string | null;
  status: "pending" | "approved" | "rejected" | string;
  visible_to_others: boolean;
  attachments: AttachmentRecord[];
  review: ReviewDecision;
  created_at: string;
};

type SendMessageResponse = {
  message_id: string | null;
  client_message_id: string;
  status: string;
  visible_to_others: boolean;
  message: MessageRecord | null;
  review: ReviewDecision;
};

type LocalAttachment = {
  id: string;
  file: File;
  url: string;
  modality: "image" | "video";
  poster?: string | null;
};

type ChatItem = MessageRecord & {
  local_attachments?: LocalAttachment[];
  review_progress?: number;
  review_started_at?: number;
};

type HealthState = "checking" | "ok" | "partial" | "down";
type ActiveView = "chat" | "audit";

const API_BASE =
  import.meta.env.VITE_GATEWAY_URL || import.meta.env.VITE_GATEWAY_BASE_URL || "";
const CONVERSATION_ID = "demo";
const CURRENT_USER = "me";
const SERVICE_NAMES: Record<string, string> = {
  text: "文字",
  image: "图片",
  video: "视频"
};
const MEDIA_REVIEW_HOLD_PROGRESS = 0.92;
const MEDIA_REVIEW_FINISH_MS = 520;

function App() {
  const [messages, setMessages] = useState<ChatItem[]>([]);
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<LocalAttachment[]>([]);
  const [sending, setSending] = useState(false);
  const [health, setHealth] = useState<HealthState>("checking");
  const [activeView, setActiveView] = useState<ActiveView>("chat");
  const [previewImage, setPreviewImage] = useState<{ src: string; name: string } | null>(
    null
  );
  const [previewVideo, setPreviewVideo] = useState<{ src: string; name: string } | null>(
    null
  );
  const [dragActive, setDragActive] = useState(false);
  const dragDepthRef = useRef(0);
  const imageInputRef = useRef<HTMLInputElement | null>(null);
  const videoInputRef = useRef<HTMLInputElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const auditScrollRef = useRef<HTMLDivElement>(null);

  const latestRejected = useMemo(
    () => [...messages].reverse().find((message) => message.status === "rejected"),
    [messages]
  );
  const rejectedMessages = useMemo(
    () => messages.filter((message) => message.status === "rejected"),
    [messages]
  );
  const hasPendingMedia = useMemo(
    () =>
      messages.some(
        (message) =>
          message.status === "pending" && (message.local_attachments?.length || 0) > 0
      ),
    [messages]
  );
  const latestMessage = messages[messages.length - 1];

  useEffect(() => {
    checkHealth();
    const timer = window.setInterval(checkHealth, 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages, activeView]);

  useEffect(() => {
    if (activeView !== "audit") {
      return;
    }
    auditScrollRef.current?.scrollTo({
      top: auditScrollRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [activeView, rejectedMessages.length]);

  useEffect(() => {
    dragDepthRef.current = 0;
    setDragActive(false);
  }, [activeView]);

  useEffect(() => {
    if (!hasPendingMedia) {
      return;
    }

    const timer = window.setInterval(() => {
      setMessages((current) => {
        const now = performance.now();
        let changed = false;
        const next = current.map((message) => {
          if (
            message.status !== "pending" ||
            !message.local_attachments?.length ||
            !message.review_started_at
          ) {
            return message;
          }

          const duration = mediaReviewEstimateMs(message.local_attachments);
          const target = Math.min(
            MEDIA_REVIEW_HOLD_PROGRESS,
            ((now - message.review_started_at) / duration) * MEDIA_REVIEW_HOLD_PROGRESS
          );
          const currentProgress = message.review_progress || 0;

          if (target <= currentProgress + 0.004) {
            return message;
          }

          changed = true;
          return {
            ...message,
            review_progress: target
          };
        });

        return changed ? next : current;
      });
    }, 80);

    return () => window.clearInterval(timer);
  }, [hasPendingMedia]);

  useEffect(() => {
    if (!previewVideo) {
      return;
    }

    function handlePreviewKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setPreviewVideo(null);
      }
    }

    window.addEventListener("keydown", handlePreviewKeyDown);
    return () => window.removeEventListener("keydown", handlePreviewKeyDown);
  }, [previewVideo]);

  useEffect(() => {
    if (attachments.length === 0) {
      return;
    }

    function handleGlobalMediaEnter(event: KeyboardEvent) {
      if (
        event.key !== "Enter" ||
        event.shiftKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.isComposing ||
        event.repeat ||
        activeView !== "chat" ||
        previewImage ||
        previewVideo ||
        sending
      ) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();
      void sendMessage();
    }

    window.addEventListener("keydown", handleGlobalMediaEnter, true);
    return () => window.removeEventListener("keydown", handleGlobalMediaEnter, true);
  }, [activeView, attachments, previewImage, previewVideo, sending, text]);

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (!response.ok) {
        setHealth("down");
        return;
      }
      const data = await response.json();
      const services = Object.values(data.services || {}) as Array<{ ok: boolean }>;
      setHealth(services.every((service) => service.ok) ? "ok" : "partial");
    } catch {
      setHealth("down");
    }
  }

  async function handlePickFiles(files: FileList | null, modality?: "image" | "video") {
    if (!files) {
      return;
    }
    const next = await Promise.all(
      Array.from(files).map(async (file) => {
        const inferred = modality || inferModality(file);
        if (!inferred) {
          return null;
        }
        const url = URL.createObjectURL(file);
        return {
          id: `local_file_${crypto.randomUUID()}`,
          file,
          url,
          modality: inferred,
          poster: inferred === "video" ? await createVideoCoverPoster(url) : null
        };
      })
    );

    const readyAttachments = next.filter(Boolean) as LocalAttachment[];
    if (readyAttachments.length === 0) {
      return;
    }
    setAttachments((current) => [...current, ...readyAttachments]);
  }

  function handleChatDragEnter(event: React.DragEvent<HTMLElement>) {
    if (activeView !== "chat" || !isFileDrag(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current += 1;
    setDragActive(true);
  }

  function handleChatDragOver(event: React.DragEvent<HTMLElement>) {
    if (activeView !== "chat" || !isFileDrag(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setDragActive(true);
  }

  function handleChatDragLeave(event: React.DragEvent<HTMLElement>) {
    if (activeView !== "chat" || !isFileDrag(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setDragActive(false);
    }
  }

  function removeAttachment(id: string) {
    setAttachments((current) => {
      const removed = current.find((item) => item.id === id);
      if (removed) {
        URL.revokeObjectURL(removed.url);
      }
      return current.filter((item) => item.id !== id);
    });
  }

  function handleDrop(event: React.DragEvent<HTMLElement>) {
    if (activeView !== "chat" || !isFileDrag(event.dataTransfer)) {
      return;
    }
    event.preventDefault();
    dragDepthRef.current = 0;
    setDragActive(false);
    void handlePickFiles(event.dataTransfer.files);
  }

  async function completePendingMediaReview(clientMessageId: string, hasMedia: boolean) {
    if (!hasMedia) {
      return;
    }

    setMessages((current) =>
      current.map((message) =>
        message.client_message_id === clientMessageId
          ? { ...message, review_progress: 1 }
          : message
      )
    );
    await sleep(MEDIA_REVIEW_FINISH_MS);
  }

  async function sendMessage() {
    const trimmed = text.trim();
    if (sending || (!trimmed && attachments.length === 0)) {
      return;
    }

    const clientMessageId = `local_${crypto.randomUUID()}`;
    const localAttachments = [...attachments];
    const pendingMessage = buildPendingMessage(
      clientMessageId,
      trimmed,
      localAttachments
    );

    setSending(true);
    setMessages((current) => [...current, pendingMessage]);
    setText("");
    setAttachments([]);

    const formData = new FormData();
    formData.append("client_message_id", clientMessageId);
    formData.append("sender_id", CURRENT_USER);
    if (trimmed) {
      formData.append("text", trimmed);
    }
    for (const attachment of localAttachments) {
      formData.append("files", attachment.file);
    }

    try {
      const response = await fetch(
        `${API_BASE}/api/conversations/${CONVERSATION_ID}/messages`,
        {
          method: "POST",
          body: formData
        }
      );
      const data = (await response.json()) as SendMessageResponse;
      if (!response.ok) {
        throw new Error("gateway_error");
      }
      const resolved = data.message
        ? normalizeMessage(data.message, localAttachments)
        : buildRejectedFallback(clientMessageId, trimmed, data.review, localAttachments);
      await completePendingMediaReview(clientMessageId, localAttachments.length > 0);
      setMessages((current) =>
        current.map((message) =>
          message.client_message_id === clientMessageId ? resolved : message
        )
      );
    } catch {
      const failed = buildRejectedFallback(
        clientMessageId,
        trimmed,
        {
          allow: false,
          status: "error",
          summary: "发送失败：审核服务暂不可用",
          reasons: []
        },
        localAttachments
      );
      await completePendingMediaReview(clientMessageId, localAttachments.length > 0);
      setMessages((current) =>
        current.map((message) =>
          message.client_message_id === clientMessageId ? failed : message
        )
      );
    } finally {
      setSending(false);
    }
  }

  function handleRetryEdit(message: ChatItem) {
    setText(message.text || "");
    setActiveView("chat");
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="rail-avatar">审</div>
        <button className="rail-button active" title="聊天">
          <MessageCircle size={22} />
        </button>
        <button className="rail-button" title="联系人">
          <User size={22} />
        </button>
        <button className="rail-button" title="设置">
          <Settings size={21} />
        </button>
      </aside>

      <aside className="conversation-pane">
        <div className="search-box">
          <Search size={16} />
          <span>搜索</span>
        </div>
        <button
          className={`conversation-item ${activeView === "chat" ? "active" : ""}`}
          onClick={() => setActiveView("chat")}
        >
          <div className="avatar">群</div>
          <div className="conversation-meta">
            <div className="conversation-line">
              <div className="conversation-title">项目小组演示群</div>
              <time>{formatShortTime(latestMessage?.created_at)}</time>
            </div>
            <div className="conversation-preview">
              {latestMessage ? messagePreview(latestMessage) : "多模态审核网关"}
            </div>
          </div>
        </button>
        <button
          className={`conversation-item ${activeView === "audit" ? "active" : ""}`}
          onClick={() => setActiveView("audit")}
        >
          <div className="avatar muted">审</div>
          <div className="conversation-meta">
            <div className="conversation-line">
              <div className="conversation-title">审核记录</div>
              {latestRejected && <time>{formatShortTime(latestRejected.created_at)}</time>}
            </div>
            <div className="conversation-preview">
              {latestRejected
                ? `${rejectedMessages.length} 条未通过：${latestRejected.review.summary}`
                : "暂无未通过记录"}
            </div>
          </div>
        </button>
      </aside>

      <main
        className={`chat-pane ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleChatDragEnter}
        onDragOver={handleChatDragOver}
        onDragLeave={handleChatDragLeave}
        onDrop={handleDrop}
      >
        <header className="chat-header">
          <div className="chat-title-block">
            <h1>{activeView === "chat" ? "项目小组演示群" : "审核记录"}</h1>
            <p>
              {activeView === "chat"
                ? `${messages.length} 条消息`
                : `${rejectedMessages.length} 条记录`}
            </p>
          </div>
          <div className="header-actions">
            <div className={`health-pill ${health}`}>
              <span />
              {healthLabel(health)}
            </div>
            <button className="header-button" title="语音">
              <Phone size={19} />
            </button>
            <button className="header-button" title="更多">
              <MoreHorizontal size={22} />
            </button>
          </div>
        </header>

        {activeView === "chat" ? (
          <>
            <section className="message-list" ref={scrollRef}>
              {messages.length === 0 ? (
                <div className="empty-state">暂无消息</div>
              ) : (
                <>
                  {messages.map((message, index) => (
                    <Fragment key={message.client_message_id || message.id}>
                      {shouldShowTime(messages, index) && (
                        <div className="time-divider">
                          {formatDividerTime(message.created_at)}
                        </div>
                      )}
                      <MessageBubble
                        message={message}
                        onPreviewImage={setPreviewImage}
                        onPreviewVideo={setPreviewVideo}
                        onRetryEdit={handleRetryEdit}
                      />
                    </Fragment>
                  ))}
                </>
              )}
            </section>

            <footer
              className="composer"
            >
              {attachments.length > 0 && (
                <div className="attachment-strip">
                  {attachments.map((attachment) => (
                    <div className="attachment-chip" key={attachment.id}>
                      <MediaThumb
                        attachment={{
                          id: attachment.id,
                          modality: attachment.modality,
                          source: attachment.url,
                          name: attachment.file.name,
                          poster: attachment.poster
                        }}
                        onPreviewImage={setPreviewImage}
                        onPreviewVideo={setPreviewVideo}
                      />
                      <span>{attachment.file.name}</span>
                      <button
                        className="icon-button"
                        title="移除"
                        onClick={() => removeAttachment(attachment.id)}
                      >
                        <X size={16} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="tool-row">
                <button className="tool-button" title="表情">
                  <Smile size={20} />
                </button>
                <button
                  className="tool-button"
                  title="图片"
                  onClick={() => imageInputRef.current?.click()}
                >
                  <ImageIcon size={20} />
                </button>
                <button
                  className="tool-button"
                  title="视频"
                  onClick={() => videoInputRef.current?.click()}
                >
                  <Video size={20} />
                </button>
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  hidden
                  onChange={(event) => {
                    void handlePickFiles(event.target.files, "image");
                    event.currentTarget.value = "";
                  }}
                />
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*"
                  hidden
                  onChange={(event) => {
                    void handlePickFiles(event.target.files, "video");
                    event.currentTarget.value = "";
                  }}
                />
              </div>
              <div className="input-row">
                <textarea
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  onKeyDown={handleKeyDown}
                  rows={3}
                />
                <button
                  className="send-button"
                  title="发送"
                  disabled={sending || (!text.trim() && attachments.length === 0)}
                  onClick={() => void sendMessage()}
                >
                  <Send size={20} />
                </button>
              </div>
            </footer>
          </>
        ) : (
          <AuditPanel
            messages={rejectedMessages}
            scrollRef={auditScrollRef}
            onRetryEdit={handleRetryEdit}
          />
        )}
      </main>

      {previewImage && (
        <div className="preview-mask" onClick={() => setPreviewImage(null)}>
          <button className="preview-close" title="关闭" onClick={() => setPreviewImage(null)}>
            <X size={24} />
          </button>
          <img
            src={previewImage.src}
            alt={previewImage.name}
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      )}

      {previewVideo && (
        <div className="preview-mask" onClick={() => setPreviewVideo(null)}>
          <button className="preview-close" title="关闭" onClick={() => setPreviewVideo(null)}>
            <X size={24} />
          </button>
          <video
            className="preview-video"
            src={previewVideo.src}
            controls
            autoPlay
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}

function MessageBubble({
  message,
  onPreviewImage,
  onPreviewVideo,
  onRetryEdit
}: {
  message: ChatItem;
  onPreviewImage: (image: { src: string; name: string }) => void;
  onPreviewVideo: (video: { src: string; name: string }) => void;
  onRetryEdit: (message: ChatItem) => void;
}) {
  const rejected = message.status === "rejected";
  const pending = message.status === "pending";
  const attachments = messageAttachments(message);
  const hasText = Boolean(message.text);
  const videoOnly =
    !hasText && attachments.length > 0 && attachments.every((attachment) => attachment.modality === "video");
  const bubbleClassName = `bubble ${videoOnly ? "video-bubble" : ""}`;

  return (
    <article className={`message-row me ${rejected ? "rejected" : ""} ${pending ? "pending" : ""}`}>
      <div className="message-stack">
        <div className={bubbleClassName}>
          {message.text && <p>{message.text}</p>}
          {attachments.length > 0 && (
            <div className="bubble-media-list">
              {attachments.map((attachment) => (
                <MediaThumb
                  attachment={attachment}
                  key={attachment.id}
                  onPreviewImage={onPreviewImage}
                  onPreviewVideo={onPreviewVideo}
                  reviewing={pending}
                  reviewProgress={message.review_progress || 0}
                />
              ))}
            </div>
          )}
        </div>
        {pending ? (
          attachments.length === 0 && (
            <div className="status-line pending">
              <Loader2 className="spin" size={14} />
              审核中
            </div>
          )
        ) : (
          <div className={`status-line ${message.status}`}>
            {rejected ? (
            <>
                <AlertTriangle size={14} />
                <span>{message.review.summary}</span>
                {message.text && (
                  <button className="inline-action" onClick={() => onRetryEdit(message)}>
                    重新编辑
                  </button>
                )}
              </>
            ) : (
              <>
                <CheckCircle2 size={14} />
                已发送
              </>
            )}
          </div>
        )}
      </div>
      <div className="mini-avatar">我</div>
    </article>
  );
}

function MediaThumb({
  attachment,
  onPreviewImage,
  onPreviewVideo,
  reviewing = false,
  reviewProgress = 0
}: {
  attachment: { id: string; modality: string; source: string; name: string; poster?: string | null };
  onPreviewImage: (image: { src: string; name: string }) => void;
  onPreviewVideo: (video: { src: string; name: string }) => void;
  reviewing?: boolean;
  reviewProgress?: number;
}) {
  return (
    <div className={`media-thumb-frame ${reviewing ? "reviewing" : ""}`}>
      {attachment.modality === "video" ? (
        <VideoThumb attachment={attachment} onPreviewVideo={onPreviewVideo} />
      ) : (
        <button
          className="image-preview-button"
          disabled={reviewing}
          onClick={() =>
            !reviewing && onPreviewImage({ src: attachment.source, name: attachment.name })
          }
          title={attachment.name}
        >
          <img src={attachment.source} alt={attachment.name} />
        </button>
      )}
      {reviewing && <MediaReviewProgress progress={reviewProgress} />}
    </div>
  );
}

function VideoThumb({
  attachment,
  onPreviewVideo
}: {
  attachment: { id: string; modality: string; source: string; name: string; poster?: string | null };
  onPreviewVideo: (video: { src: string; name: string }) => void;
}) {
  const generatedPoster = useVideoCoverPoster(attachment.source, !attachment.poster);
  const poster = attachment.poster || generatedPoster;

  return (
    <button
      className="video-cover-button"
      title="播放视频"
      onClick={() => onPreviewVideo({ src: attachment.source, name: attachment.name })}
    >
      {poster ? (
        <img className="video-cover-image" src={poster} alt={attachment.name} />
      ) : (
        <span className="video-cover-placeholder" aria-hidden="true" />
      )}
      <span className="video-preview-button">
        <Play size={18} fill="currentColor" />
      </span>
    </button>
  );
}

function useVideoCoverPoster(source: string, enabled = true) {
  const [poster, setPoster] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setPoster(null);

    if (enabled) {
      void createVideoCoverPoster(source).then((nextPoster) => {
        if (!cancelled) {
          setPoster(nextPoster);
        }
      });
    }

    return () => {
      cancelled = true;
    };
  }, [source, enabled]);

  return poster;
}

async function createVideoCoverPoster(source: string): Promise<string | null> {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";

  const waitFor = (eventName: string, timeoutMs = 1600) =>
    new Promise<void>((resolve) => {
      const timer = window.setTimeout(resolve, timeoutMs);
      video.addEventListener(
        eventName,
        () => {
          window.clearTimeout(timer);
          resolve();
        },
        { once: true }
      );
    });

  const captureFrame = (): { dataUrl: string; luminance: number } | null => {
    if (!video.videoWidth || !video.videoHeight) {
      return null;
    }

    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / video.videoWidth);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) {
      return null;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let totalLuminance = 0;
    let count = 0;
    for (let index = 0; index < pixels.length; index += 4 * 24) {
      totalLuminance +=
        pixels[index] * 0.2126 +
        pixels[index + 1] * 0.7152 +
        pixels[index + 2] * 0.0722;
      count += 1;
    }

    return {
      dataUrl: canvas.toDataURL("image/jpeg", 0.84),
      luminance: count > 0 ? totalLuminance / count : 0
    };
  };

  const seekTo = async (time: number) => {
    if (!Number.isFinite(time)) {
      return;
    }
    const duration = Number.isFinite(video.duration) ? video.duration : time;
    video.currentTime = Math.min(Math.max(time, 0.05), Math.max(0.05, duration - 0.05));
    await waitFor("seeked");
  };

  try {
    video.src = source;
    video.load();
    await waitFor("loadedmetadata");
    if (!video.videoWidth || !video.videoHeight) {
      return null;
    }

    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const candidates = Array.from(
      new Set(
        [0.2, 0.8, Math.min(2, duration * 0.25), duration * 0.5]
          .filter((time) => Number.isFinite(time) && time > 0)
          .map((time) => Number(time.toFixed(2)))
      )
    );
    if (candidates.length === 0) {
      candidates.push(0.05);
    }

    let bestFrame: { dataUrl: string; luminance: number } | null = null;
    for (const time of candidates) {
      await seekTo(time);
      const frame = captureFrame();
      if (!frame) {
        continue;
      }
      if (!bestFrame || frame.luminance > bestFrame.luminance) {
        bestFrame = frame;
      }
      if (frame.luminance > 34) {
        break;
      }
    }

    return bestFrame?.dataUrl || null;
  } catch {
    return null;
  } finally {
    video.removeAttribute("src");
    video.load();
  }
}

function MediaReviewProgress({ progress }: { progress: number }) {
  const clampedProgress = Math.min(1, Math.max(0, progress));
  const style = {
    "--media-review-progress": `${clampedProgress * 360}deg`
  } as CSSProperties;

  return (
    <div className="media-review-overlay" aria-label="审核中">
      <span className="media-review-progress" style={style} />
    </div>
  );
}

function AuditPanel({
  messages,
  scrollRef,
  onRetryEdit
}: {
  messages: ChatItem[];
  scrollRef: RefObject<HTMLDivElement>;
  onRetryEdit: (message: ChatItem) => void;
}) {
  return (
    <section className="audit-panel">
      <div className="audit-feed" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="empty-state audit-empty">暂无未通过记录</div>
        ) : (
          messages.map((message, index) => (
            <AuditNoticeCard
              key={message.id}
              message={message}
              index={index}
              onRetryEdit={onRetryEdit}
            />
          ))
        )}
      </div>
      <footer className="audit-footer">
        {messages.length === 0 ? "暂无审核记录" : `共 ${messages.length} 条未通过记录`}
      </footer>
    </section>
  );
}

function AuditNoticeCard({
  message,
  index,
  onRetryEdit
}: {
  message: ChatItem;
  index: number;
  onRetryEdit: (message: ChatItem) => void;
}) {
  const reasons = reviewReasons(message);
  const attachments = messageAttachments(message);
  const content = message.text || attachmentSummary(attachments);

  return (
    <article
      className="audit-notice-card"
      style={{ animationDelay: `${Math.min(index * 45, 270)}ms` }}
    >
      <div className="notice-header">
        <div className="notice-icon">
          <AlertTriangle size={18} />
        </div>
        <div className="notice-title-block">
          <div className="notice-title">内容审核通知</div>
          <time>{formatDividerTime(message.created_at)}</time>
        </div>
        <span className="notice-status">未发送</span>
      </div>

      <div className="notice-body">
        <p className="notice-summary">{message.review.summary}</p>
        {content && <p className="notice-content">{content}</p>}
      </div>

      <div className="notice-reasons">
        {reasons.map((reason, index) => (
          <div className="notice-reason" key={`${reason.modality}-${reason.code}-${index}`}>
            <span>{SERVICE_NAMES[reason.modality] || reason.modality}</span>
            <strong>{reason.display}</strong>
            {typeof reason.score === "number" && (
              <em>{(reason.score * 100).toFixed(1)}%</em>
            )}
          </div>
        ))}
      </div>

      {message.text && (
        <div className="notice-actions">
          <button className="notice-action-button" onClick={() => onRetryEdit(message)}>
            重新编辑
          </button>
        </div>
      )}
    </article>
  );
}

function reviewReasons(message: ChatItem): ReviewReason[] {
  if (message.review.reasons.length > 0) {
    return message.review.reasons;
  }
  return [
    {
      modality: "system",
      code: "unknown",
      display: message.review.summary
    }
  ];
}

function attachmentSummary(
  attachments: Array<{ id: string; modality: string; source: string; name: string }>
) {
  if (attachments.length === 0) {
    return "";
  }
  const imageCount = attachments.filter((attachment) => attachment.modality === "image").length;
  const videoCount = attachments.filter((attachment) => attachment.modality === "video").length;
  const parts = [];
  if (imageCount > 0) {
    parts.push(`${imageCount} 张图片`);
  }
  if (videoCount > 0) {
    parts.push(`${videoCount} 个视频`);
  }
  return parts.length > 0 ? parts.join("、") : `${attachments.length} 个附件`;
}

function normalizeMessage(
  message: MessageRecord,
  localAttachments: LocalAttachment[] = []
): ChatItem {
  return {
    ...message,
    local_attachments: localAttachments.length > 0 ? localAttachments : undefined
  };
}

function mediaReviewEstimateMs(localAttachments: LocalAttachment[]) {
  return localAttachments.some((attachment) => attachment.modality === "video")
    ? 12000
    : 1800;
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function buildPendingMessage(
  clientMessageId: string,
  text: string,
  localAttachments: LocalAttachment[]
): ChatItem {
  return {
    id: clientMessageId,
    client_message_id: clientMessageId,
    conversation_id: CONVERSATION_ID,
    sender_id: CURRENT_USER,
    text: text || null,
    status: "pending",
    visible_to_others: false,
    attachments: [],
    local_attachments: localAttachments,
    review: {
      allow: false,
      status: "reviewing",
      summary: "审核中",
      reasons: []
    },
    created_at: new Date().toISOString(),
    review_progress: localAttachments.length > 0 ? 0 : undefined,
    review_started_at: localAttachments.length > 0 ? performance.now() : undefined
  };
}

function buildRejectedFallback(
  clientMessageId: string,
  text: string,
  review: ReviewDecision,
  localAttachments: LocalAttachment[]
): ChatItem {
  return {
    id: clientMessageId,
    client_message_id: clientMessageId,
    conversation_id: CONVERSATION_ID,
    sender_id: CURRENT_USER,
    text,
    status: "rejected",
    visible_to_others: false,
    attachments: [],
    local_attachments: localAttachments,
    review,
    created_at: new Date().toISOString()
  };
}

function healthLabel(health: HealthState) {
  if (health === "ok") {
    return "服务在线";
  }
  if (health === "partial") {
    return "部分在线";
  }
  if (health === "down") {
    return "服务离线";
  }
  return "检查中";
}

function messagePreview(message?: ChatItem) {
  if (!message) {
    return "多模态审核网关";
  }
  if (message.status === "rejected") {
    return message.review.summary;
  }
  if (message.text) {
    return message.text;
  }
  if (message.attachments.some((attachment) => attachment.modality === "video")) {
    return "[视频]";
  }
  if (message.attachments.length > 0) {
    return "[图片]";
  }
  return "已发送";
}

function messageAttachments(message: ChatItem) {
  return (
    message.local_attachments?.map((attachment) => ({
      id: attachment.id,
      modality: attachment.modality,
      source: attachment.url,
      name: attachment.file.name,
      poster: attachment.poster
    })) ||
    message.attachments.map((attachment) => ({
      id: attachment.id,
      modality: attachment.modality,
      source: mediaSource(attachment.storage_path),
      name: attachment.original_name
    }))
  );
}

function isFileDrag(dataTransfer: DataTransfer) {
  return Array.from(dataTransfer.types).includes("Files");
}

function inferModality(file: File): "image" | "video" | null {
  if (file.type.startsWith("image/")) {
    return "image";
  }
  if (file.type.startsWith("video/")) {
    return "video";
  }

  const extension = file.name.split(".").pop()?.toLowerCase();
  if (
    extension &&
    ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff"].includes(extension)
  ) {
    return "image";
  }
  if (
    extension &&
    ["mp4", "mov", "m4v", "webm", "avi", "mkv"].includes(extension)
  ) {
    return "video";
  }
  return null;
}

function shouldShowTime(messages: ChatItem[], index: number) {
  if (index === 0) {
    return true;
  }
  const current = new Date(messages[index].created_at).getTime();
  const previous = new Date(messages[index - 1].created_at).getTime();
  return Number.isFinite(current) && Number.isFinite(previous)
    ? current - previous > 5 * 60 * 1000
    : false;
}

function formatDividerTime(value?: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function formatShortTime(value?: string) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function mediaSource(storagePath: string) {
  const marker = "storage/uploads/";
  const normalized = storagePath.replace(/\\/g, "/");
  const index = normalized.indexOf(marker);
  const relative = index >= 0 ? normalized.slice(index + marker.length) : normalized;
  return `${API_BASE}/media/uploads/${relative
    .split("/")
    .map((part: string) => encodeURIComponent(part))
    .join("/")}`;
}

export default App;
