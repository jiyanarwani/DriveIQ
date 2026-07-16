import React, { useState } from 'react'
import axios from 'axios'

export default function LoginPanel({ onLogin, onClose }) {
  const [isRegister, setIsRegister] = useState(false)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    const endpoint = isRegister ? '/api/auth/register' : '/api/auth/login'
    try {
      const res = await axios.post(endpoint, { email, password })
      if (res.data.token) onLogin(res.data.token)
    } catch (err) {
      setError(err.response?.data?.error || 'Authentication failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-header">
          <span className="modal-title">
            {isRegister ? 'Create Account' : 'Sign In'}
          </span>
          <button className="modal-close" onClick={onClose} aria-label="Close">x</button>
        </div>

        {error && <div className="form-error mb-2">{error}</div>}

        <form onSubmit={handleSubmit} className="form-group">
          <div>
            <label className="label" htmlFor="auth-email">Email</label>
            <input
              id="auth-email"
              className="input"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />
          </div>
          <div>
            <label className="label" htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              className="input"
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="password"
              required
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={loading}
          >
            {loading ? 'Working...' : isRegister ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        <div className="modal-footer">
          <button className="link-btn" type="button" onClick={() => setIsRegister(!isRegister)}>
            {isRegister ? 'Already have an account? Sign in.' : 'Need an account? Register.'}
          </button>
        </div>
      </div>
    </div>
  )
}
