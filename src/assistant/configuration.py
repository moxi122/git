import os
from dataclasses import dataclass, fields
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

# 默认报告结构：作为 prompt 中的 {report_structure} 注入，决定最终报告的章节骨架
DEFAULT_REPORT_STRUCTURE = """
# Introduction
- Brief overview of the research topic or question.
- Purpose and scope of the report.

# Main Body
- For each section (1-4 sections):
  - Subheading: Provide a relevant subheading to the section's key aspect.
  - Explanation: A detailed explanation of the concept or topic being discussed in the section.
  - Findings/Details: Support the explanation with research findings, statistics, examples, or case studies.

# Key Takeaways
- Bullet points summarizing the most important insights or findings.

# Conclusion
- Final summary of the research.
- Implications or relevance of the findings.   
"""

# 可配置项：这三个字段既能从 config["configurable"] 传入，也能用同名大写环境变量覆盖
@dataclass(kw_only=True)
class Configuration:
    """研究代理的可配置字段。"""
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    max_search_queries: int = 5
    enable_web_search: bool = False

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从 RunnableConfig 构造 Configuration 实例：环境变量优先，其次 configurable，取到非空值才生效。"""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
