# Description

The model server supports optional per-process rate admission before upstream
requests. `request_rate_limit` reserves request count and a conservative token
budget over `period_seconds` (60 by default). Both Responses and Chat calls
share the same budget. Configure each process's share of the provider quota;
this does not coordinate different processes or accounts.

```yaml
request_rate_limit:
  requests_per_period: 4000
  tokens_per_period: 21000000
  period_seconds: 60
  output_token_reservation: 32768
  admission_timeout_seconds: 300
```

The input reservation counts serialized UTF-8 bytes plus framing overhead.
`output_token_reservation` declares output capacity for quota accounting when
the request has no explicit output limit; it does not change generation.
Each HTTP attempt, including a transport retry, reserves both budgets. Failed
or cancelled attempts keep their reservations until expiry.
Waiting requests have a finite deadline and can be cancelled. No request body
is truncated, and retries remain owned by the existing HTTP transport.
Omitting `request_rate_limit` preserves the default behavior.

OpenAI-compatible model server using Gym's shared HTTP client.

## HTTP retries

HTTP 404 and 408 are retryable by default, alongside the existing transient
server and rate-limit errors. This applies to every dataset and to policy and
judge clients. A permanent 404 still fails after the bounded attempt budget.
Retries resend the same request with the existing fixed 0.5-second delay
between attempts and preserve the terminal error body.

`max_http_attempts` defaults to three total attempts. Set it per model server:

```yaml
judge_model:
  responses_api_models:
    openai_model:
      max_http_attempts: 5
```

A value of one disables HTTP retries. Connection-error retries are controlled
separately. Internal Gym clients retain their existing unbounded extension for
rate-limit statuses; 404 and 408 never trigger that extension.


# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0

With admission enabled, model POST requests disable automatic HTTP redirects; redirects are returned without following them. Non-POST client operations reject before dispatch because aiohttp can transparently retry idempotent methods. Unrestricted admission preserves the existing redirect and method behavior.
