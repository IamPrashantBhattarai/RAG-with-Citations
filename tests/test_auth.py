"""Sign-in settings (Phase 13): absent means local mode; broken means stop, never open."""

import pytest
import streamlit
from streamlit import config as st_config
from streamlit.runtime.secrets import Secrets

from citerag import auth

FILLED = """\
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "a-long-random-cookie-secret"
client_id = "123456789012-abc.apps.googleusercontent.com"
client_secret = "GOCSPX-not-a-real-secret"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
"""


@pytest.fixture
def secrets_file(monkeypatch, tmp_path):
    """Point Streamlit's real secrets loader at a temporary secrets.toml; write(None) leaves no file."""
    path = tmp_path / "secrets.toml"
    old = st_config.get_option("secrets.files")
    st_config.set_option("secrets.files", [str(path)])
    monkeypatch.setattr(Secrets, "_maybe_install_file_watchers", lambda self: None)
    monkeypatch.setattr(streamlit, "secrets", Secrets())  # fresh, so nothing is cached between tests

    def write(text):
        if text is not None:
            path.write_text(text)
        return path

    yield write
    st_config.set_option("secrets.files", old)


def test_no_secrets_file_means_local_mode(secrets_file):
    secrets_file(None)
    assert auth.configured() is False


@pytest.mark.parametrize("text", [
    FILLED.replace("123456789012-abc", "PASTE-your-client-id").replace("GOCSPX-not-a-real-secret",
                                                                       "PASTE-your-client-secret"),
    "[server_settings]\nsomething = 1\n",  # no [auth] section
])
def test_sign_in_never_set_up_means_local_mode(secrets_file, text):
    secrets_file(text)
    assert auth.configured() is False


def test_filled_in_settings_turn_sign_in_on(secrets_file):
    secrets_file(FILLED)
    assert auth.configured() is True


def test_an_unreadable_secrets_file_stops_the_app_instead_of_opening_it(secrets_file, tmp_path):
    # The real mistake: pasting the client id over the placeholder lost its closing quote.
    secrets_file(FILLED.replace('apps.googleusercontent.com"', "apps.googleusercontent.com"))
    with pytest.raises(auth.AuthConfigError) as error:
        auth.configured()
    message = str(error.value)
    assert "can't be read" in message and "line 4" in message  # says where to look
    assert str(tmp_path) not in message and "GOCSPX" not in message  # no path or secret on the page


@pytest.mark.parametrize("change, named", [
    (("GOCSPX-not-a-real-secret", "PASTE-your-client-secret"), "client_secret"),  # only the id pasted
    (('client_id = "123456789012-abc.apps.googleusercontent.com"', 'client_id = ""'), "client_id"),
    (('cookie_secret = "a-long-random-cookie-secret"\n', ""), "cookie_secret"),  # a line deleted
])
def test_a_partly_filled_in_setup_stops_the_app_and_names_what_is_missing(secrets_file, change, named):
    secrets_file(FILLED.replace(*change))
    with pytest.raises(auth.AuthConfigError, match=f"partly set up; fill in {named}"):
        auth.configured()


def test_auth_must_be_a_section(secrets_file):
    secrets_file('auth = "google"\n')
    with pytest.raises(auth.AuthConfigError, match="must be a section"):
        auth.configured()
