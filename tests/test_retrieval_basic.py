from __future__ import annotations
import os
from guardian_policy_agent.db.session import init_engine, get_session
from guardian_policy_agent.db.models import Base, PolicyDoc, PolicyStatement, PolicySection
from guardian_policy_agent.retrieval.hybrid import hybrid_rank

def setup_module():
    os.environ["DB_URL"] = "sqlite:///./test_step6.db"
    init_engine(os.environ["DB_URL"], False)
    from guardian_policy_agent.db.session import get_engine
    eng = get_engine()
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)

    # seed one doc + statements
    with get_session() as ses:
        d = PolicyDoc(doc_id="example.com#privacy_unknown", domain="example.com")
        ses.add(d)
        ses.add(PolicySection(policy_doc_id=d.doc_id, sid="S1", title="Main"))
        ses.add(PolicyStatement(
            policy_doc_id=d.doc_id, sid="S1",
            data_categories=["online_identifiers","cookies_tech"],
            actions=["store","use"], purposes=["service_delivery"],
            recipients=[], legal_basis=[],
            evidence_snippet="We use cookies to store your preferences."
        ))
        ses.add(PolicyStatement(
            policy_doc_id=d.doc_id, sid="S1",
            data_categories=["identifiers"], actions=["collect"],
            purposes=["ads_first_party"], evidence_snippet="Used for ads."
        ))
        ses.commit()

def test_hybrid_rank_simple():
    from guardian_policy_agent.db.session import get_session
    with get_session() as ses:
        stmts = ses.query(PolicyStatement).all()
        objs = []
        for s in stmts:
            objs.append({
                "evidence_id": f"{s.policy_doc_id}:{s.id}",
                "policy_doc_id": s.policy_doc_id,
                "domain": "example.com",
                "section": s.sid,
                "data_categories": s.data_categories,
                "actions": s.actions,
                "purposes": s.purposes,
                "recipients": s.recipients,
                "legal_basis": s.legal_basis,
                "snippet": s.evidence_snippet,
            })
        behavior = {
            "platform":"web","domain":"example.com","app_id":None,"action_type":"xhr",
            "data_categories":["online_identifiers"],"actions":["use"],"purposes":["service_delivery"],"recipients":[]
        }
        ranked = hybrid_rank(objs, behavior, alpha=0.7)
        assert len(ranked) == len(objs)
        # the cookies/service_delivery stmt should appear first
        assert ranked[0][0] in (0,1)
