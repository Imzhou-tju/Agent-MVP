import os
import json
import threading
import queue
import atexit
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────────────────────
# 全行为审计日志器
#
# 架构：生产者-消费者模型，主链路只做 queue.put()，不阻塞在磁盘 IO。
#   前台线程（主循环）  ──put──▶  log_queue（内存）  ◀──get──  后台守护线程（写盘）
#
# 每条事件落一行 JSONL，格式统一：
#   {"ts": "2026-07-09T08:12:01Z", "thread_id": "xxx", "event": "xxx", ...字段}
#
# 4 类核心事件（由 agent.py 埋点产生）：
#   llm_input   : 主循环即将调用 LLM 前，记录本次打包的消息条数
#                 {"ts":..., "event":"llm_input", "message_count": 6}
#
#   tool_call   : LLM 返回 tool_calls 时，逐个记录工具名和入参
#                 {"ts":..., "event":"tool_call", "tool":"search_academic",
#                  "args":{"query":"Mamba SSM","max_results":5}}
#
#   tool_result : ToolNode 执行完毕后，下一轮 agent_node 回看末尾 tool
#                 message 时补记（非 hook 执行瞬间，而是下一轮进入时补录）
#                 {"ts":..., "event":"tool_result", "tool":"search_academic",
#                  "result_summary":"[{\"title\":\"Mamba...\",\"url\":\"...\"}]"}
#
#   ai_message  : LLM 直接回复用户（未调工具）时记录完整内容
#                 {"ts":..., "event":"ai_message", "content":"Mamba 是..."}
#
# 日志文件按 thread_id 分文件写入：logs/<thread_id>.jsonl
# 监控终端（entry/monitor.py）tail -f 该文件实时渲染。
# ─────────────────────────────────────────────────────────────────────────────
class JSONLEventLogger:
    # 单例：整个进程共享一个日志器实例，避免多实例竞争写同一文件
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = "logs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_logger(log_dir)
            return cls._instance

    def _init_logger(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 无界内存队列：前台 put 永不阻塞，后台消费者异步落盘
        self.log_queue = queue.Queue()

        # daemon=True：主进程退出时守护线程自动销毁，atexit 负责先 flush
        self.worker_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.worker_thread.start()

        # 程序正常退出时先等队列写完再退，防止末尾事件丢失
        atexit.register(self.shutdown)

    def _write_loop(self):
        """后台消费者：阻塞等队列，有事件就写盘，收到 None 哨兵则退出。"""
        while True:
            log_item = self.log_queue.get()

            if log_item is None:  # shutdown() 发送的哨兵，表示退出信号
                self.log_queue.task_done()
                break

            try:
                thread_id = log_item.get("thread_id", "system")
                # 过滤非法文件名字符，防止路径注入
                safe_id = "".join(c for c in thread_id if c.isalnum() or c in "-_") or "default"
                file_path = os.path.join(self.log_dir, f"{safe_id}.jsonl")

                # 追加写入，每条事件一行 JSON（JSONL 格式）
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Logger Error] 异步写日志失败: {e}")
            finally:
                self.log_queue.task_done()

    def log_event(self, thread_id: str, event: str, **kwargs):
        """前台埋点方法：拼装事件对象后丢进内存队列，立即返回不阻塞主链路。

        event 取值范围：llm_input | tool_call | tool_result | ai_message
        **kwargs 按事件类型携带不同字段，见类头注释。
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log_item = {
            "ts": now_utc,           # ISO 8601 UTC 时间戳
            "thread_id": thread_id,  # 对话会话 ID，对应日志文件名
            "event": event,          # 事件类型
            **kwargs                 # 事件专属字段（message_count/tool/args/content 等）
        }

        self.log_queue.put(log_item)  # O(1)，不等磁盘

    def shutdown(self):
        """发送 None 哨兵，等后台线程把队列里剩余事件全部写完再退出。"""
        self.log_queue.put(None)
        self.log_queue.join()

audit_logger = JSONLEventLogger()