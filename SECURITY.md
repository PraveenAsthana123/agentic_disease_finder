# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.5.x   | :white_check_mark: |
| 2.4.x   | :white_check_mark: |
| < 2.4   | :x:                |

## Reporting a Vulnerability

We take security seriously, especially given the healthcare/clinical nature of this project.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers directly
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- Acknowledgment within 48 hours
- Status update within 7 days
- Fix timeline based on severity:
  - **Critical**: 24-48 hours
  - **High**: 7 days
  - **Medium**: 30 days
  - **Low**: Next release

## Security Considerations for Healthcare AI

### PHI/PII Protection

This framework processes potentially sensitive health data. Users must:

1. **Never commit real patient data** to the repository
2. **De-identify all data** before processing
3. **Use encryption** for data at rest and in transit
4. **Implement access controls** per HIPAA/GDPR requirements

### Model Security

| Risk | Mitigation |
|------|------------|
| Model extraction | Rate limiting, access logging |
| Adversarial inputs | Input validation, OOD detection |
| Data poisoning | Data provenance tracking |
| Model inversion | Differential privacy |

### Secure Deployment Checklist

- [ ] All PHI removed from training data
- [ ] Encryption enabled (TLS 1.3+)
- [ ] Access controls configured (RBAC)
- [ ] Audit logging enabled
- [ ] Vulnerability scanning passed
- [ ] Penetration testing completed
- [ ] Incident response plan documented

## Dependencies

We regularly scan dependencies for vulnerabilities using:
- `safety check`
- `pip-audit`
- Dependabot alerts

## Compliance

This project is designed with the following regulations in mind:
- HIPAA (US)
- GDPR (EU)
- PHIPA (Ontario, Canada)
- FDA SaMD guidance

**Note**: Compliance is the responsibility of the deploying organization.
