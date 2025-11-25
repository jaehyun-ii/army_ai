/**
 * Password validation utilities for client-side validation.
 *
 * Implements Korean military password policy:
 * - Minimum 9 characters with numbers, letters, and special characters
 * - Cannot be same as username
 * - Cannot contain personal/department information
 * - Cannot use common dictionary words
 * - No 3+ consecutive identical characters
 * - No ascending/descending sequences
 */

export interface PasswordValidationResult {
  isValid: boolean
  errors: string[]
}

export class PasswordValidator {
  // Common Korean dictionary words
  private static readonly COMMON_WORDS = new Set([
    'password', 'admin', 'user', 'test', 'qwerty',
    '비밀번호', '관리자', '사용자', '테스트',
    'army', 'military', 'soldier', 'korea',
    '육군', '군대', '병사', '한국', '대한민국'
  ])

  // Personal/department related keywords to block
  private static readonly BLOCKED_KEYWORDS = new Set([
    '생년월일', '전화번호', '주민번호', 'birthday', 'phone',
    '부대', '사단', '여단', '연대', '대대', '중대', '소대', 'division', 'brigade',
    '이름', 'name', '성명'
  ])

  /**
   * Validate password against all security rules.
   */
  static validatePassword(password: string, username?: string): PasswordValidationResult {
    const errors: string[] = []

    // Rule 1: Minimum 9 characters with mixed types
    if (!this.checkLengthAndComplexity(password)) {
      errors.push('비밀번호는 숫자, 문자, 특수문자를 포함하여 9자리 이상이어야 합니다.')
    }

    // Rule 2: Cannot be same as username
    if (username && password.toLowerCase() === username.toLowerCase()) {
      errors.push('비밀번호는 사용자 ID와 동일할 수 없습니다.')
    }

    // Rule 3: Cannot contain username
    if (username && password.toLowerCase().includes(username.toLowerCase())) {
      errors.push('비밀번호에 사용자 ID가 포함될 수 없습니다.')
    }

    // Rule 4: Cannot contain personal/department keywords
    if (this.containsBlockedKeywords(password)) {
      errors.push('비밀번호에 개인정보 또는 부서명칭이 포함될 수 없습니다.')
    }

    // Rule 5: Cannot be common dictionary word
    if (this.isCommonWord(password)) {
      errors.push('일반 사전에 등록된 단어는 사용할 수 없습니다.')
    }

    // Rule 6: No 3+ consecutive identical characters
    if (this.hasRepeatedChars(password)) {
      errors.push('동일한 문자를 3회 이상 연속으로 사용할 수 없습니다.')
    }

    // Rule 7: No ascending/descending sequences
    if (this.hasSequentialChars(password)) {
      errors.push('연속적인 오름차순 또는 내림차순 문자열은 사용할 수 없습니다.')
    }

    return {
      isValid: errors.length === 0,
      errors
    }
  }

  /**
   * Check if password meets length and complexity requirements.
   */
  private static checkLengthAndComplexity(password: string): boolean {
    if (password.length < 9) {
      return false
    }

    const hasNumber = /\d/.test(password)
    const hasLetter = /[a-zA-Z]/.test(password)
    const hasSpecial = /[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;/~`]/.test(password)

    return hasNumber && hasLetter && hasSpecial
  }

  /**
   * Check if password contains blocked personal/department keywords.
   */
  private static containsBlockedKeywords(password: string): boolean {
    const passwordLower = password.toLowerCase()
    return Array.from(this.BLOCKED_KEYWORDS).some(keyword =>
      passwordLower.includes(keyword.toLowerCase())
    )
  }

  /**
   * Check if password is a common dictionary word.
   */
  private static isCommonWord(password: string): boolean {
    return this.COMMON_WORDS.has(password.toLowerCase())
  }

  /**
   * Check for 3+ consecutive identical characters.
   */
  private static hasRepeatedChars(password: string): boolean {
    for (let i = 0; i < password.length - 2; i++) {
      if (password[i] === password[i + 1] && password[i] === password[i + 2]) {
        return true
      }
    }
    return false
  }

  /**
   * Check for ascending or descending sequences (3+ chars).
   */
  private static hasSequentialChars(password: string): boolean {
    if (password.length < 3) {
      return false
    }

    for (let i = 0; i < password.length - 2; i++) {
      const char1 = password.charCodeAt(i)
      const char2 = password.charCodeAt(i + 1)
      const char3 = password.charCodeAt(i + 2)

      // Check ascending sequence (e.g., abc, 123)
      if (char2 === char1 + 1 && char3 === char2 + 1) {
        return true
      }

      // Check descending sequence (e.g., cba, 321)
      if (char2 === char1 - 1 && char3 === char2 - 1) {
        return true
      }
    }

    return false
  }

  /**
   * Get real-time validation feedback for password input.
   */
  static getPasswordStrength(password: string, username?: string): {
    strength: 'weak' | 'medium' | 'strong'
    checks: {
      length: boolean
      complexity: boolean
      noUsername: boolean
      noRepeated: boolean
      noSequential: boolean
    }
  } {
    const checks = {
      length: password.length >= 9,
      complexity: this.checkLengthAndComplexity(password),
      noUsername: !username || password.toLowerCase() !== username.toLowerCase(),
      noRepeated: !this.hasRepeatedChars(password),
      noSequential: !this.hasSequentialChars(password)
    }

    const passedChecks = Object.values(checks).filter(Boolean).length
    const strength = passedChecks <= 2 ? 'weak' : passedChecks <= 4 ? 'medium' : 'strong'

    return { strength, checks }
  }
}

/**
 * Convenience function for password validation.
 */
export function validatePassword(password: string, username?: string): PasswordValidationResult {
  return PasswordValidator.validatePassword(password, username)
}
