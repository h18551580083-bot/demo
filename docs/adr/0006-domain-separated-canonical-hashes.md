# Use domain-separated SHA-256 over RFC 8785 envelopes

Scientific and derived identities use domain-separated SHA-256 envelopes whose
metadata is true RFC 8785 JCS and whose tensor data is a canonical raw byte
payload. This is stricter than ordinary sorted JSON and requires explicit handling
of duplicate keys, unsafe numbers, Unicode code points, negative zero, and payload
lengths, but it makes parameter, kernel, execution, and cache identities
reproducible across compliant implementations and prevents cross-domain byte
collisions.
