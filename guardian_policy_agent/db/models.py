from __future__ import annotations
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, DateTime, Text, Boolean, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import JSONB

class Base(DeclarativeBase):
    pass

# Use JSONB in Postgres for performant operators and GIN indexing.
JSON_COMPAT = JSON().with_variant(JSONB, "postgresql")

class PolicyDoc(Base):
    __tablename__ = "policy_docs"
    doc_id: Mapped[str] = mapped_column(String(256), primary_key=True)          # e.g., "bing.com#privacy_2024-01-01"
    domain: Mapped[str] = mapped_column(String(255), index=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    version_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    policy_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    raw_refs: Mapped[Optional[dict]] = mapped_column(JSON_COMPAT)                # paths to html/txt
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    sections: Mapped[List["PolicySection"]] = relationship(back_populates="doc", cascade="all,delete-orphan")
    statements: Mapped[List["PolicyStatement"]] = relationship(back_populates="doc", cascade="all,delete-orphan")

class PolicySection(Base):
    __tablename__ = "policy_sections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_doc_id: Mapped[str] = mapped_column(ForeignKey("policy_docs.doc_id", ondelete="CASCADE"))
    sid: Mapped[str] = mapped_column(String(32))                                 # e.g., "S1"
    title: Mapped[Optional[str]] = mapped_column(String(255))

    doc: Mapped["PolicyDoc"] = relationship(back_populates="sections")

class PolicyStatement(Base):
    __tablename__ = "policy_statements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_doc_id: Mapped[str] = mapped_column(ForeignKey("policy_docs.doc_id", ondelete="CASCADE"), index=True)
    sid: Mapped[str] = mapped_column(String(32))
    data_categories: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)         # JSON/JSONB arrays
    actions: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    purposes: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    recipients: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    legal_basis: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)

    retention_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "period"/"criteria"/None
    retention_value: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    transfer_outside_eea_uk: Mapped[bool] = mapped_column(Boolean, default=False)
    rights_flag: Mapped[bool] = mapped_column(Boolean, default=False)

    evidence_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    evidence_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column()

    doc: Mapped["PolicyDoc"] = relationship(back_populates="statements")

Index(
    "ix_policy_statements_gin_data",
    PolicyStatement.data_categories,
    postgresql_using="gin",
)
Index(
    "ix_policy_statements_gin_actions",
    PolicyStatement.actions,
    postgresql_using="gin",
)
Index(
    "ix_policy_statements_gin_purposes",
    PolicyStatement.purposes,
    postgresql_using="gin",
)

class Behavior(Base):
    """
    Canonicalized user/app behavior definition (from monitor or pre-defined behavior catalog).
    """
    __tablename__ = "behaviors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    app_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    platform: Mapped[Optional[str]] = mapped_column(String(32))                  # web/mobile/desktop
    data_categories: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    actions: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    purposes: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    recipients: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    meta: Mapped[Optional[dict]] = mapped_column(JSON_COMPAT)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class PolicyLink(Base):
    """
    Result of rule/linking (e.g., obey_violate) between behavior and policy statements/doc.
    """
    __tablename__ = "policy_links"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_doc_id: Mapped[str] = mapped_column(ForeignKey("policy_docs.doc_id", ondelete="CASCADE"), index=True)
    behavior_id: Mapped[int] = mapped_column(ForeignKey("behaviors.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(16))                              # obey/violate/conditional/unknown
    evidence: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)                # [{sid, snippet}, ...]

class UserPref(Base):
    """
    User consent/preference key-values (scoped).
    """
    __tablename__ = "user_prefs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    namespace: Mapped[str] = mapped_column(String(64), default="consent")        # e.g., consent, ui, privacy
    key: Mapped[str] = mapped_column(String(128))
    value: Mapped[dict] = mapped_column(JSON_COMPAT)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class MonitorEvent(Base):
    """
    Raw event captured by Monitor (normalized).
    """
    __tablename__ = "monitor_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    platform: Mapped[str] = mapped_column(String(32))                            # web/mobile
    domain: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    app_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    action_type: Mapped[Optional[str]] = mapped_column(String(128))              # e.g., clipboard_read, xhr_send, permission_request
    data_categories: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    actions: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    purposes: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    recipients: Mapped[Optional[list]] = mapped_column(JSON_COMPAT)
    event_metadata: Mapped[Optional[dict]] = mapped_column(JSON_COMPAT)
    # metadata: Mapped[Optional[dict]] = mapped_column(JSON)                      # request path hash, headers, etc.

class Decision(Base):
    """
    Policy Agent decision per event.
    """
    __tablename__ = "decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("monitor_events.id", ondelete="CASCADE"), index=True)
    decision: Mapped[str] = mapped_column(String(16))                            # allow/deny/transform/ask
    risk_score: Mapped[Optional[float]] = mapped_column()
    payload: Mapped[Optional[dict]] = mapped_column(JSON_COMPAT)                 # full LLM JSON output
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class DecisionEvidence(Base):
    """
    Links decision to policy statements used as evidence.
    """
    __tablename__ = "decision_evidence"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(ForeignKey("decisions.id", ondelete="CASCADE"), index=True)
    policy_statement_id: Mapped[int] = mapped_column(ForeignKey("policy_statements.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(32), default="support")             # support/contradict/neutral
    note: Mapped[Optional[str]] = mapped_column(Text)

class UsageMetric(Base):
    """Privacy-preserving product telemetry. Request bodies are never stored."""
    __tablename__ = "usage_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    endpoint: Mapped[str] = mapped_column(String(64), index=True)
    install_id_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    status_code: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    decision: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sensitive: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    categories: Mapped[Optional[list]] = mapped_column(JSON_COMPAT, nullable=True)
    initial_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    final_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    upgraded: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
