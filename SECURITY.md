# Security Policy

## Supported Versions

Currently supported version with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Updates

### Recent Security Patches (2026-01-16)

We have updated all dependencies to address known security vulnerabilities:

#### 1. python-multipart (CVE)
- **Issue**: Denial of service (DoS) via deformation `multipart/form-data` boundary
- **Affected**: < 0.0.18
- **Fixed**: Updated to **0.0.18**
- **Impact**: Prevents DoS attacks through malformed form data

#### 2. PyTorch (CVE)
- **Issue**: `torch.load` with `weights_only=True` leads to remote code execution
- **Affected**: < 2.6.0
- **Fixed**: Updated to **2.6.0**
- **Impact**: Prevents RCE when loading model weights

#### 3. HuggingFace Transformers (CVE)
- **Issue**: Deserialization of Untrusted Data in Hugging Face Transformers (3 instances)
- **Affected**: >= 0, < 4.48.0
- **Fixed**: Updated to **4.48.0**
- **Impact**: Prevents untrusted data deserialization attacks

### Security Best Practices

This application follows security best practices:

1. **Dependency Management**: All dependencies are pinned to specific versions
2. **Regular Updates**: Dependencies are updated to address security vulnerabilities
3. **Input Validation**: File upload validation for type and size
4. **CORS Configuration**: Restricted to specific origins (localhost in development)
5. **Code Scanning**: CodeQL security scanning enabled
6. **Type Safety**: TypeScript for frontend type checking

## Reporting a Vulnerability

If you discover a security vulnerability in this project:

1. **Do Not** open a public issue
2. Email the maintainer directly (if provided)
3. Or create a private security advisory on GitHub
4. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Security Scanning

This project uses:
- **CodeQL** for static code analysis
- **npm audit** for frontend dependency scanning
- **pip-audit** (recommended) for backend dependency scanning

### Running Security Scans

**Frontend:**
```bash
cd frontend
npm audit
npm audit fix  # Fix automatically if possible
```

**Backend:**
```bash
cd backend
pip install pip-audit
pip-audit
```

## Security Headers

When deploying to production, ensure these security headers are configured:

- `Content-Security-Policy`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Strict-Transport-Security` (if using HTTPS)
- `Referrer-Policy: no-referrer`

## API Security

### File Upload Protection

The API includes protection against:
- **Malicious File Types**: Only image files accepted
- **Large Files**: Implement size limits (recommended: < 10MB)
- **Path Traversal**: Use safe file handling

### Model Loading

- Models are loaded from trusted sources (HuggingFace)
- Weights are cached locally
- No arbitrary code execution from model files (PyTorch 2.6.0+)

## Production Deployment

For production deployments:

1. **Use HTTPS**: Always use TLS/SSL
2. **Environment Variables**: Store secrets in environment variables, not code
3. **Rate Limiting**: Implement rate limiting on API endpoints
4. **Authentication**: Add authentication for production use
5. **Monitoring**: Set up security monitoring and logging
6. **Firewall**: Configure firewall rules appropriately
7. **Updates**: Keep dependencies updated regularly

## Dependencies with Known Vulnerabilities

✅ **All resolved** as of 2026-01-16

Previous vulnerabilities (now fixed):
- ~~python-multipart 0.0.12~~ → Updated to 0.0.18
- ~~torch 2.5.1~~ → Updated to 2.6.0
- ~~transformers 4.46.3~~ → Updated to 4.48.0

## Security Checklist for Deployment

- [ ] All dependencies updated to latest secure versions
- [ ] HTTPS enabled
- [ ] CORS properly configured
- [ ] File upload size limits enforced
- [ ] Input validation on all endpoints
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Logging and monitoring set up
- [ ] Regular security audits scheduled
- [ ] Incident response plan in place

## Contact

For security concerns, please follow responsible disclosure practices.

---

**Last Updated**: 2026-01-16
**Security Level**: ✅ No known vulnerabilities
