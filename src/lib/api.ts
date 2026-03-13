const API_URL = process.env.NEXT_PUBLIC_API_URL || "";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": API_KEY,
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }

  return res.json();
}

export interface AnalyseResult {
  task: string;
  prediction_label?: string;
  confidence?: number;
  risk_level?: string;
  entities?: { token: string; label: string }[];
}

export interface FeedbackResult {
  status: string;
  message: string;
  stats: LearningStats;
}

export interface LearningStats {
  feedback_buffered: number;
  total_learned: number;
  learning_rate: number;
  batch_size: number;
}

export function analyse(text: string, task: string): Promise<AnalyseResult> {
  return request("/analyse", {
    method: "POST",
    body: JSON.stringify({ text, task }),
  });
}

export function submitFeedback(
  text: string,
  task: string,
  correct_label: string
): Promise<FeedbackResult> {
  return request("/feedback", {
    method: "POST",
    body: JSON.stringify({ text, task, correct_label }),
  });
}

export function getLearningStats(): Promise<LearningStats> {
  return request("/learning-stats");
}
