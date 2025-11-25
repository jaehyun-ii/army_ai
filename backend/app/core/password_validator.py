"""
Password validation utilities for security compliance.

Implements Korean military password policy:
- Minimum 9 characters with numbers, letters, and special characters
- Cannot be same as username
- Cannot contain personal/department information
- Cannot use common dictionary words
- No 3+ consecutive identical characters
- No ascending/descending sequences
"""
import re
from typing import List, Tuple


class PasswordValidator:
    """Password validation for security compliance."""

    # Common Korean dictionary words (can be extended)
    COMMON_WORDS = {
    }

    # Personal/department related keywords to block
    BLOCKED_KEYWORDS = {
    }

    @staticmethod
    def validate_password(password: str, username: str = None) -> Tuple[bool, List[str]]:
        """
        Validate password against all security rules.

        Args:
            password: Password to validate
            username: Username to check against (optional)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Rule 1: Minimum 9 characters with mixed types
        if not PasswordValidator._check_length_and_complexity(password):
            errors.append("비밀번호는 숫자, 문자, 특수문자를 포함하여 9자리 이상이어야 합니다.")

        # Rule 2: Cannot be same as username
        if username and password.lower() == username.lower():
            errors.append("비밀번호는 사용자 ID와 동일할 수 없습니다.")

        # Rule 3: Cannot contain username
        if username and username.lower() in password.lower():
            errors.append("비밀번호에 사용자 ID가 포함될 수 없습니다.")

        # Rule 4: Cannot contain personal/department keywords
        if PasswordValidator._contains_blocked_keywords(password):
            errors.append("비밀번호에 개인정보 또는 부서명칭이 포함될 수 없습니다.")

        # Rule 5: Cannot be common dictionary word
        if PasswordValidator._is_common_word(password):
            errors.append("일반 사전에 등록된 단어는 사용할 수 없습니다.")

        # Rule 6: No 3+ consecutive identical characters
        if PasswordValidator._has_repeated_chars(password):
            errors.append("동일한 문자를 3회 이상 연속으로 사용할 수 없습니다.")

        # Rule 7: No ascending/descending sequences
        if PasswordValidator._has_sequential_chars(password):
            errors.append("연속적인 오름차순 또는 내림차순 문자열은 사용할 수 없습니다.")

        return len(errors) == 0, errors

    @staticmethod
    def _check_length_and_complexity(password: str) -> bool:
        """Check if password meets length and complexity requirements."""
        if len(password) < 9:
            return False

        has_number = bool(re.search(r'\d', password))
        has_letter = bool(re.search(r'[a-zA-Z]', password))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/~`]', password))

        return has_number and has_letter and has_special

    @staticmethod
    def _contains_blocked_keywords(password: str) -> bool:
        """Check if password contains blocked personal/department keywords."""
        password_lower = password.lower()
        return any(keyword in password_lower for keyword in PasswordValidator.BLOCKED_KEYWORDS)

    @staticmethod
    def _is_common_word(password: str) -> bool:
        """Check if password is a common dictionary word."""
        password_lower = password.lower()
        return password_lower in PasswordValidator.COMMON_WORDS

    @staticmethod
    def _has_repeated_chars(password: str) -> bool:
        """Check for 3+ consecutive identical characters."""
        for i in range(len(password) - 2):
            if password[i] == password[i + 1] == password[i + 2]:
                return True
        return False

    @staticmethod
    def _has_sequential_chars(password: str) -> bool:
        """
        Check for ascending or descending sequences (3+ chars).
        Checks for sequences in numbers and letters.
        """
        if len(password) < 3:
            return False

        for i in range(len(password) - 2):
            # Get ASCII values
            char1 = ord(password[i])
            char2 = ord(password[i + 1])
            char3 = ord(password[i + 2])

            # Check ascending sequence (e.g., abc, 123)
            if char2 == char1 + 1 and char3 == char2 + 1:
                return True

            # Check descending sequence (e.g., cba, 321)
            if char2 == char1 - 1 and char3 == char2 - 1:
                return True

        return False


# Convenience function
def validate_password(password: str, username: str = None) -> Tuple[bool, List[str]]:
    """
    Validate password against all security rules.

    Args:
        password: Password to validate
        username: Username to check against (optional)

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    return PasswordValidator.validate_password(password, username)
