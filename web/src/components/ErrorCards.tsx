import { ErrorDetail } from "../lib/types";

const ERROR_PRESENTATION: Record<string, { title: string; tone: string; defaultHint: string }> = {
  missing_index: {
    title: "Index not found",
    tone: "danger",
    defaultHint: "Set SECURITYCLIP_INDEX to a directory containing securityclip.sqlite and restart Security Scope."
  },
  missing_api_key: {
    title: "OpenAI API key missing",
    tone: "danger",
    defaultHint: "Set OPENAI_API_KEY (and OPENAI_BASE_URL if your provider needs one) and restart Security Scope."
  },
  router_failed: {
    title: "Query routing failed",
    tone: "warn",
    defaultHint: "Retry the query; if it persists, check the router model name and API connectivity."
  },
  planner_failed: {
    title: "Operation planning failed",
    tone: "warn",
    defaultHint: "Retry the query; if it persists, check the planner model name and API connectivity."
  },
  validation_failed: {
    title: "Plan validation issue",
    tone: "warn",
    defaultHint: "The planner emitted an invalid operation; it was adjusted or replaced with a fallback search."
  },
  repair_failed: {
    title: "Plan repair failed",
    tone: "warn",
    defaultHint: "The planner could not repair its invalid output; a fallback search was used."
  },
  operation_timeout: {
    title: "Operation timed out",
    tone: "warn",
    defaultHint: "Increase SECURITYCLIP_COMMAND_TIMEOUT or narrow the operation scope."
  },
  operation_failed: {
    title: "Operation failed",
    tone: "warn",
    defaultHint: "Check the command trace for the failing operation."
  },
  no_results: {
    title: "No results",
    tone: "info",
    defaultHint: "Try broader terms or remove source filters."
  },
  synthesis_failed: {
    title: "Answer synthesis failed",
    tone: "warn",
    defaultHint: "The answer shown is a deterministic fallback; retry for a model-written report."
  }
};

export default function ErrorCards({ details, legacyErrors }: { details?: ErrorDetail[]; legacyErrors?: string[] }) {
  if (details && details.length > 0) {
    return (
      <div className="error-cards">
        {details.map((detail, idx) => {
          const meta = ERROR_PRESENTATION[detail.code] ?? { title: detail.code, tone: "warn", defaultHint: "" };
          const hint = detail.hint || meta.defaultHint;
          return (
            <div key={`${detail.code}-${idx}`} className={`error-card tone-${meta.tone}`}>
              <strong>{meta.title}</strong>
              <p>{detail.message}</p>
              {hint && <p className="error-hint">{hint}</p>}
            </div>
          );
        })}
      </div>
    );
  }
  if (legacyErrors && legacyErrors.length > 0) {
    return (
      <div className="warning-block">
        {legacyErrors.map((item) => (
          <p key={item}>{item}</p>
        ))}
      </div>
    );
  }
  return null;
}
