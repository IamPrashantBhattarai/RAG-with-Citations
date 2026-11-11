"""Sign-in with Google, through Streamlit's built-in OpenID Connect support. (Phase 13)

Google proves who the user is; CiteRAG stores no passwords. Settings go in
.streamlit/secrets.toml (git-ignored), section [auth]: redirect_uri,
cookie_secret, client_id, client_secret, server_metadata_url. Until the client
id and secret are filled in, sign-in is off and the app runs as a local,
single-user tool, as before.

Who is let in, and what they see, is decided by citerag.access (access.toml),
not here: here we only say which verified email address is signed in.
"""

from __future__ import annotations

REQUIRED = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
PLACEHOLDER = "PASTE"  # the template's client id and secret start with this until filled in


def settings() -> dict:
    import streamlit as st

    try:
        return dict(st.secrets.get("auth", {}))
    except Exception:  # no secrets.toml at all
        return {}


def configured() -> bool:
    """True when every sign-in setting is present and filled in (not the template's placeholder)."""
    s = settings()
    return all(str(s.get(k, "")).strip() and not str(s.get(k, "")).startswith(PLACEHOLDER) for k in REQUIRED)


def identity() -> str | None:
    """The signed-in user's email, if Google verified it; otherwise None."""
    import streamlit as st

    user = st.user
    if not getattr(user, "is_logged_in", False):
        return None
    email = user.get("email")
    if not email or user.get("email_verified") is False:
        return None  # an address Google hasn't verified proves nothing
    return str(email).strip().lower()


def sign_in() -> None:
    import streamlit as st

    st.login()


def sign_out() -> None:
    import streamlit as st

    st.logout()
