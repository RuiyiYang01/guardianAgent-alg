from __future__ import annotations
import os
from guardian_policy_agent.db.session import init_engine, get_session
from guardian_policy_agent.db.models import Base, MonitorEvent, UserPref

def setup_module():
    os.environ["DB_URL"] = "sqlite:///./test_step6.db"
    init_engine(os.environ["DB_URL"], False)
    from guardian_policy_agent.db.session import get_engine
    eng = get_engine()
    Base.metadata.create_all(eng)

def test_rule_fallback_ads_optout_allows_deny():
    from guardian_policy_agent.service.decider import decide_for_event
    with get_session() as ses:
        # prefs: ads opt-out
        ses.add(UserPref(user_id="u1", namespace="consent", key="ads", value="opt-out"))
        # event: ads purpose present
        ev = MonitorEvent(user_id="u1", platform="web", domain="example.com",
                          action_type="fetch", purposes=["ads_first_party"], actions=["use"], data_categories=[])
        ses.add(ev); ses.commit()
        out = decide_for_event(ses, ev.id, use_llm=False)
        assert out["decision"] == "deny"
