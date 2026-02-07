"""Tests for the authentication module."""

import pytest
from datetime import timedelta
from jose import jwt

from sie_x.api.auth import (
    create_access_token,
    verify_password,
    get_password_hash,
    SECRET_KEY,
    ALGORITHM,
    FAKE_USERS_DB,
)


class TestPasswordHashing:

    def test_verify_correct_password(self):
        hashed = get_password_hash("secret123")
        assert verify_password("secret123", hashed) is True

    def test_verify_wrong_password(self):
        hashed = get_password_hash("secret123")
        assert verify_password("wrong", hashed) is False

    def test_hash_is_different_from_plaintext(self):
        hashed = get_password_hash("mypassword")
        assert hashed != "mypassword"


class TestAccessToken:

    def test_create_token_returns_string(self):
        token = create_access_token(data={"sub": "testuser"})
        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_contains_subject(self):
        token = create_access_token(data={"sub": "alice", "role": "admin"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "alice"
        assert payload["role"] == "admin"

    def test_token_has_expiry(self):
        token = create_access_token(
            data={"sub": "bob"},
            expires_delta=timedelta(minutes=5),
        )
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload

    def test_expired_token_raises(self):
        token = create_access_token(
            data={"sub": "expired"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(Exception):
            jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    def test_invalid_token_raises(self):
        with pytest.raises(Exception):
            jwt.decode("not.a.token", SECRET_KEY, algorithms=[ALGORITHM])


class TestFakeUsersDB:

    def test_admin_exists(self):
        assert "admin" in FAKE_USERS_DB
        admin = FAKE_USERS_DB["admin"]
        assert admin.username == "admin"
        assert verify_password("admin", admin.hashed_password)

    def test_user_exists(self):
        assert "user" in FAKE_USERS_DB
        user = FAKE_USERS_DB["user"]
        assert verify_password("user", user.hashed_password)
