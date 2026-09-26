"""本地只读 Dashboard：把 logs/*.jsonl 的审计事件增量索引进 SQLite，供查询与可视化。

数据边界（对应 spec §8/§9）：
- JSONL 是唯一审计数据源；SQLite 只做查询索引，不变成第二套业务状态。
- 增量索引：只读新增行，不整包加载全部历史日志到内存。
- Dashboard 只读，不改 Research State；查询异常不影响调研主流程。
"""

from .indexer import AuditIndexer, build_index, index_events
from .server import DashboardServer

__all__ = ["AuditIndexer", "DashboardServer", "build_index", "index_events"]
