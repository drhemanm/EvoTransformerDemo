"use client";

import { useState } from "react";
import {
  analyse,
  submitFeedback,
  getLearningStats,
  getAnalytics,
  type AnalyseResult,
  type LearningStats,
  type AnalyticsData,
} from "@/lib/api";

const TASKS = ["transaction", "document", "ner"] as const;
type Task = (typeof TASKS)[number];

const TASK_LABELS: Record<Task, string[]> = {
  transaction: [
    "education",
    "entertainment",
    "food_grocery",
    "healthcare",
    "salary_income",
    "transfer",
    "transport",
    "utilities",
  ],
  document: [
    "business_operations",
    "financial_statement",
    "legal_regulatory",
    "management_governance",
    "risk_disclosure",
  ],
  ner: [
    "O",
    "B-ORG",
    "I-ORG",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-MONEY",
    "I-MONEY",
  ],
};

const TASK_DESCRIPTIONS: Record<Task, string> = {
  transaction: "Classify financial transactions by category and risk level",
  document: "Classify compliance documents by type",
  ner: "Extract named entities (organizations, people, locations, money)",
};

// ─── Reusable components ─────────────────────────────────────

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-gray-800 bg-gray-900 p-6 ${className}`}>
      <h2 className="mb-4 text-lg font-semibold text-white">{title}</h2>
      {children}
    </div>
  );
}

function Badge({
  children,
  variant = "default",
}: {
  children: React.ReactNode;
  variant?: "default" | "success" | "warning" | "info";
}) {
  const colors = {
    default: "bg-gray-800 text-gray-300",
    success: "bg-emerald-900/50 text-emerald-400 border border-emerald-800",
    warning: "bg-amber-900/50 text-amber-400 border border-amber-800",
    info: "bg-blue-900/50 text-blue-400 border border-blue-800",
  };
  return (
    <span
      className={`inline-block rounded-full px-3 py-1 text-xs font-medium ${colors[variant]}`}
    >
      {children}
    </span>
  );
}

// ─── Landing / Hero Section ─────────────────────────────────

function HeroSection({ onGetStarted }: { onGetStarted: () => void }) {
  return (
    <div className="mb-12 text-center">
      <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-blue-600/20 border border-blue-500/30">
        <svg className="h-8 w-8 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
        </svg>
      </div>
      <h1 className="mb-3 text-4xl font-bold text-white">EvoCompliance</h1>
      <p className="mx-auto mb-6 max-w-2xl text-lg text-gray-400">
        AI-powered compliance intelligence that learns from your feedback.
        Classify transactions, analyse documents, and extract entities — all from
        a single adaptive model.
      </p>

      <div className="mx-auto mb-8 grid max-w-3xl gap-4 md:grid-cols-3">
        <FeatureCard
          title="Transaction Classification"
          description="Categorize financial transactions into 8 categories with automated risk scoring"
          icon="T"
        />
        <FeatureCard
          title="Document Analysis"
          description="Classify compliance documents across 5 regulatory categories"
          icon="D"
        />
        <FeatureCard
          title="Entity Extraction"
          description="Identify organizations, people, locations, and monetary amounts"
          icon="E"
        />
      </div>

      <div className="mx-auto max-w-2xl rounded-xl border border-blue-800/50 bg-blue-900/20 p-4 mb-6">
        <p className="text-sm text-blue-300">
          <strong>What makes this different?</strong> EvoCompliance uses online learning —
          submit corrections and watch the model improve in real time. No retraining pipelines,
          no waiting. Your feedback directly shapes the model.
        </p>
      </div>

      <button
        onClick={onGetStarted}
        className="rounded-lg bg-blue-600 px-8 py-3 text-sm font-medium text-white transition hover:bg-blue-500"
      >
        Get Started
      </button>
    </div>
  );
}

function FeatureCard({ title, description, icon }: { title: string; description: string; icon: string }) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5 text-left">
      <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600/20 text-sm font-bold text-blue-400">
        {icon}
      </div>
      <h3 className="mb-1 text-sm font-semibold text-white">{title}</h3>
      <p className="text-xs text-gray-400">{description}</p>
    </div>
  );
}

// ─── Analyse Panel ───────────────────────────────────────────

function AnalysePanel() {
  const [text, setText] = useState("");
  const [task, setTask] = useState<Task>("transaction");
  const [result, setResult] = useState<AnalyseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAnalyse() {
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await analyse(text, task);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card title="Analyse">
      <p className="mb-4 text-sm text-gray-400">
        {TASK_DESCRIPTIONS[task]}
      </p>

      {/* Task selector */}
      <div className="mb-4 flex gap-2">
        {TASKS.map((t) => (
          <button
            key={t}
            onClick={() => {
              setTask(t);
              setResult(null);
              setError("");
            }}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              task === t
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Text input */}
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          task === "transaction"
            ? "e.g. Monthly salary deposit from employer..."
            : task === "document"
              ? "e.g. The board approved the annual risk assessment report..."
              : "e.g. Apple Inc. paid $1.5 billion to John Smith in London..."
        }
        rows={4}
        className="mb-4 w-full rounded-lg border border-gray-700 bg-gray-800 p-3 text-sm text-gray-100 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      />

      <button
        onClick={handleAnalyse}
        disabled={loading || !text.trim()}
        className="rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Analysing..." : "Analyse"}
      </button>

      {/* Error */}
      {error && (
        <div className="mt-4 rounded-lg border border-red-800 bg-red-900/30 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="mt-4 space-y-3">
          {result.task === "ner" ? (
            <NerResult entities={result.entities || []} />
          ) : (
            <ClassificationResult result={result} />
          )}
        </div>
      )}
    </Card>
  );
}

function ClassificationResult({ result }: { result: AnalyseResult }) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <div className="flex items-center gap-3">
        <span className="text-sm text-gray-400">Prediction:</span>
        <Badge variant="info">{result.prediction_label}</Badge>
        {result.risk_level && (
          <Badge
            variant={result.risk_level === "low" ? "success" : "warning"}
          >
            Risk: {result.risk_level}
          </Badge>
        )}
      </div>
      {result.confidence !== undefined && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-xs text-gray-400">
            <span>Confidence</span>
            <span>{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-700">
            <div
              className="h-2 rounded-full bg-blue-500 transition-all"
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function NerResult({ entities }: { entities: { token: string; label: string }[] }) {
  const labelColors: Record<string, string> = {
    "B-ORG": "bg-purple-900/50 text-purple-300 border-purple-700",
    "I-ORG": "bg-purple-900/30 text-purple-400 border-purple-800",
    "B-PER": "bg-emerald-900/50 text-emerald-300 border-emerald-700",
    "I-PER": "bg-emerald-900/30 text-emerald-400 border-emerald-800",
    "B-LOC": "bg-orange-900/50 text-orange-300 border-orange-700",
    "I-LOC": "bg-orange-900/30 text-orange-400 border-orange-800",
    "B-MONEY": "bg-yellow-900/50 text-yellow-300 border-yellow-700",
    "I-MONEY": "bg-yellow-900/30 text-yellow-400 border-yellow-800",
    O: "bg-transparent text-gray-400 border-transparent",
  };

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
      <div className="mb-3 flex flex-wrap gap-1.5 text-xs">
        {["ORG", "PER", "LOC", "MONEY"].map((type) => (
          <span
            key={type}
            className={`rounded px-2 py-0.5 border ${labelColors[`B-${type}`]}`}
          >
            {type}
          </span>
        ))}
      </div>
      <div className="flex flex-wrap gap-1">
        {entities.map((e, i) => (
          <span
            key={i}
            className={`rounded border px-1.5 py-0.5 text-sm ${labelColors[e.label] || labelColors.O}`}
            title={e.label}
          >
            {e.token.replace("##", "")}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Feedback Panel ──────────────────────────────────────────

function FeedbackPanel() {
  const [text, setText] = useState("");
  const [task, setTask] = useState<Task>("transaction");
  const [label, setLabel] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSubmit() {
    if (!text.trim() || !label) return;
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const res = await submitFeedback(text, task, label);
      setMessage(res.message);
      setText("");
      setLabel("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card title="Feedback">
      <p className="mb-4 text-sm text-gray-400">
        Correct the model by submitting the right label for a given text. The
        model learns from your feedback in real time.
      </p>

      {/* Task selector */}
      <div className="mb-3 flex gap-2">
        {TASKS.map((t) => (
          <button
            key={t}
            onClick={() => {
              setTask(t);
              setLabel("");
              setError("");
              setMessage("");
            }}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              task === t
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter the text that was misclassified..."
        rows={3}
        className="mb-3 w-full rounded-lg border border-gray-700 bg-gray-800 p-3 text-sm text-gray-100 placeholder-gray-500 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
      />

      {/* Label selector */}
      <div className="mb-4">
        <label className="mb-2 block text-xs font-medium text-gray-400">
          Correct label
        </label>
        <select
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2.5 text-sm text-gray-100 focus:border-blue-500 focus:outline-none"
        >
          <option value="">Select label...</option>
          {TASK_LABELS[task].map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading || !text.trim() || !label}
        className="rounded-lg bg-emerald-600 px-6 py-2.5 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? "Submitting..." : "Submit Feedback"}
      </button>

      {error && (
        <div className="mt-3 rounded-lg border border-red-800 bg-red-900/30 p-3 text-sm text-red-400">
          {error}
        </div>
      )}
      {message && (
        <div className="mt-3 rounded-lg border border-emerald-800 bg-emerald-900/30 p-3 text-sm text-emerald-400">
          {message}
        </div>
      )}
    </Card>
  );
}

// ─── Stats Panel ─────────────────────────────────────────────

function StatsPanel() {
  const [stats, setStats] = useState<LearningStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleRefresh() {
    setLoading(true);
    setError("");
    try {
      const res = await getLearningStats();
      setStats(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card title="Learning Stats">
      <p className="mb-4 text-sm text-gray-400">
        Monitor the online learning system.
      </p>

      <button
        onClick={handleRefresh}
        disabled={loading}
        className="mb-4 rounded-lg bg-gray-700 px-5 py-2 text-sm font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
      >
        {loading ? "Loading..." : "Refresh Stats"}
      </button>

      {error && (
        <div className="rounded-lg border border-red-800 bg-red-900/30 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {stats && (
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Buffered" value={stats.feedback_buffered} />
          <StatCard label="Total Learned" value={stats.total_learned} />
          <StatCard label="Learning Rate" value={stats.learning_rate} />
          <StatCard label="Batch Size" value={stats.batch_size} />
        </div>
      )}
    </Card>
  );
}

// ─── Analytics Dashboard ─────────────────────────────────────

function AnalyticsDashboard() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleRefresh() {
    setLoading(true);
    setError("");
    try {
      const res = await getAnalytics();
      setData(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card title="Usage Analytics" className="md:col-span-2">
      <p className="mb-4 text-sm text-gray-400">
        Track API usage, model performance, and learning progress.
      </p>

      <button
        onClick={handleRefresh}
        disabled={loading}
        className="mb-4 rounded-lg bg-gray-700 px-5 py-2 text-sm font-medium text-gray-200 transition hover:bg-gray-600 disabled:opacity-50"
      >
        {loading ? "Loading..." : "Load Analytics"}
      </button>

      {error && (
        <div className="mb-4 rounded-lg border border-red-800 bg-red-900/30 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-4">
          {/* Summary row */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatCard label="Requests (1h)" value={data.last_hour.total} />
            <StatCard label="Requests (24h)" value={data.last_24h.total} />
            <StatCard
              label="Avg Confidence"
              value={data.last_24h.avg_confidence !== null
                ? `${(data.last_24h.avg_confidence * 100).toFixed(1)}%`
                : "N/A"}
            />
            <StatCard label="Total Learned" value={data.learning.total_learned} />
          </div>

          {/* Task breakdown */}
          {data.last_24h.total > 0 && (
            <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
              <h3 className="mb-3 text-sm font-medium text-gray-300">Requests by Task (24h)</h3>
              <div className="space-y-2">
                {Object.entries(data.last_24h.by_task).map(([task, count]) => (
                  <div key={task} className="flex items-center gap-3">
                    <span className="w-24 text-xs text-gray-400">{task}</span>
                    <div className="flex-1">
                      <div className="h-2 w-full rounded-full bg-gray-700">
                        <div
                          className="h-2 rounded-full bg-blue-500 transition-all"
                          style={{
                            width: `${(count / data.last_24h.total) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                    <span className="w-8 text-right text-xs text-gray-400">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Source breakdown */}
          {data.last_24h.total > 0 && (
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-300">Analyses</h3>
                <div className="text-2xl font-bold text-blue-400">
                  {data.last_24h.by_source.analyse || 0}
                </div>
              </div>
              <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4">
                <h3 className="mb-2 text-sm font-medium text-gray-300">Feedback</h3>
                <div className="text-2xl font-bold text-emerald-400">
                  {data.last_24h.by_source.feedback || 0}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function StatCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-3 text-center">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="mt-1 text-xs text-gray-400">{label}</div>
    </div>
  );
}

// ─── Navigation Tabs ─────────────────────────────────────────

type TabId = "home" | "analyse" | "dashboard";

function NavTabs({ active, onChange }: { active: TabId; onChange: (tab: TabId) => void }) {
  const tabs: { id: TabId; label: string }[] = [
    { id: "home", label: "Home" },
    { id: "analyse", label: "Analyse" },
    { id: "dashboard", label: "Dashboard" },
  ];

  return (
    <div className="mb-8 flex gap-1 rounded-lg bg-gray-900 p-1 border border-gray-800">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition ${
            active === tab.id
              ? "bg-blue-600 text-white"
              : "text-gray-400 hover:text-white hover:bg-gray-800"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabId>("home");

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <NavTabs active={activeTab} onChange={setActiveTab} />

      {activeTab === "home" && (
        <HeroSection onGetStarted={() => setActiveTab("analyse")} />
      )}

      {activeTab === "analyse" && (
        <div className="space-y-6">
          <AnalysePanel />
          <div className="grid gap-6 md:grid-cols-2">
            <FeedbackPanel />
            <StatsPanel />
          </div>
        </div>
      )}

      {activeTab === "dashboard" && (
        <div className="space-y-6">
          <AnalyticsDashboard />
        </div>
      )}

      {/* Footer */}
      <footer className="mt-12 border-t border-gray-800 pt-6 text-center text-xs text-gray-500">
        EvoCompliance v0.3.0 — Powered by EvoTransformer with online learning
      </footer>
    </div>
  );
}
