"""Sign-in with Google, through Streamlit's built-in OpenID Connect support. (Phase 13)

Google proves who the user is; CiteRAG stores no passwords. Settings go in
.streamlit/secrets.toml (git-ignored), section [auth]: redirect_uri,
cookie_secret, client_id, client_secret, server_metadata_url. Until the client
id and secret are filled in, sign-in is off and the app runs as a local,
single-user tool, as before.

A setup that is broken rather than absent fails closed: a secrets file that
can't be read, or an [auth] section that is only partly filled in, raises
AuthConfigError, and the app stops instead of opening without sign-in.

Who is let in, and what they see, is decided by citerag.access (access.toml),
not here: here we only say which verified email address is signed in.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

log = logging.getLogger(__name__)

REQUIRED = ("redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url")
CLIENT = ("client_id", "client_secret")  # both unset means sign-in was never set up
PLACEHOLDER = "PASTE"  # the template's client id and secret start with this until filled in
NO_SECRETS_FILE = "no-secrets-found"  # Streamlit's error_id when no secrets file exists at all


class AuthConfigError(Exception):
    """The sign-in settings exist but are broken. The app must stop, not open without sign-in."""


def settings() -> dict:
    """The [auth] settings; {} when there is no secrets file or no [auth] section.

    A secrets file that exists but can't be read raises AuthConfigError. Treating it
    like "no settings" would quietly turn sign-in off and open the app to anyone.
    """
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    try:
        section = st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError as e:
        if getattr(e, "error_id", None) == NO_SECRETS_FILE:
            return {}
        # Streamlit's message includes the file's full path; the TOML error alone says what to fix.
        detail = e.__cause__ or e
        log.error("Sign-in settings can't be read: %s", e)
        raise AuthConfigError(f".streamlit/secrets.toml can't be read ({detail})") from None
    except Exception as e:  # anything unexpected: closed, never open
        log.error("Sign-in settings can't be read: %s", e)
        raise AuthConfigError(f".streamlit/secrets.toml can't be read ({type(e).__name__})") from None
    if not isinstance(section, Mapping):
        raise AuthConfigError("[auth] in .streamlit/secrets.toml must be a section of settings")
    return dict(section)


def _unset(value) -> bool:
    text = str(value if value is not None else "").strip()
    return not text or text.startswith(PLACEHOLDER)


def configured() -> bool:
    """Whether sign-in is on.

    False only when sign-in was never set up: no secrets file, no [auth] section, or
    the client id and secret both still unset (the template as created). Raises
    AuthConfigError for anything in between, such as an unreadable file or only
    some settings filled in.
    """
    s = settings()
    if all(_unset(s.get(k)) for k in CLIENT):
        return False
    missing = [k for k in REQUIRED if _unset(s.get(k))]
    if missing:
        raise AuthConfigError(f"sign-in is only partly set up; fill in {', '.join(missing)} under [auth]")
    return True


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
