"use client";

type CausalEdge = {
  id: string;
  from: string;
  to: string;
  evidence_tool_call_id?: string;
};

type CausalGraphData = {
  edges: CausalEdge[];
  confidence?: number;
  confidence_rationale?: string;
  validation_error?: string;
};

export function CausalGraph({ graph }: { graph: CausalGraphData }) {
  if (!graph || !graph.edges || graph.edges.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center text-sm text-[var(--muted)]">
        {graph?.validation_error
          ? `No causal graph produced: ${graph.validation_error}`
          : "No causal graph produced for this incident."}
      </div>
    );
  }

  const nodes: string[] = [];
  for (const edge of graph.edges) {
    if (!nodes.includes(edge.from)) nodes.push(edge.from);
    if (!nodes.includes(edge.to)) nodes.push(edge.to);
  }

  const nodeWidth = 200;
  const nodeHeight = 56;
  const gapX = 90;
  const svgWidth = nodes.length * nodeWidth + (nodes.length - 1) * gapX + 40;
  const svgHeight = 160;
  const centerY = svgHeight / 2;

  const nodeX = (i: number) => 20 + i * (nodeWidth + gapX);

  return (
    <div className="space-y-3">
      {typeof graph.confidence === "number" && (
        <div className="text-sm text-[var(--foreground)]">
          Confidence:{" "}
          <span
            className={
              graph.confidence >= 0.6 ? "font-medium text-[var(--accent-teal)]" : "font-medium text-[var(--accent-amber)]"
            }
          >
            {(graph.confidence * 100).toFixed(0)}%
          </span>
          {graph.confidence_rationale && (
            <span className="ml-2 text-[var(--muted)]">— {graph.confidence_rationale}</span>
          )}
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
        <svg width={svgWidth} height={svgHeight} className="min-w-full">
          {graph.edges.map((edge, i) => {
            const fromIdx = nodes.indexOf(edge.from);
            const toIdx = nodes.indexOf(edge.to);
            const x1 = nodeX(fromIdx) + nodeWidth;
            const x2 = nodeX(toIdx);
            const hasEvidence = Boolean(edge.evidence_tool_call_id);
            return (
              <g key={edge.id ?? i}>
                <line
                  x1={x1}
                  y1={centerY}
                  x2={x2}
                  y2={centerY}
                  stroke={hasEvidence ? "#2dd4bf" : "#f59e0b"}
                  strokeWidth={2}
                  markerEnd="url(#arrowhead)"
                />
                {hasEvidence && (
                  <text
                    x={(x1 + x2) / 2}
                    y={centerY - 10}
                    textAnchor="middle"
                    className="fill-[#8896ab] text-[10px]"
                  >
                    {edge.evidence_tool_call_id}
                  </text>
                )}
              </g>
            );
          })}

          <defs>
            <marker
              id="arrowhead"
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M0,0 L8,4 L0,8 Z" fill="#2dd4bf" />
            </marker>
          </defs>

          {nodes.map((node, i) => (
            <g key={node}>
              <rect
                x={nodeX(i)}
                y={centerY - nodeHeight / 2}
                width={nodeWidth}
                height={nodeHeight}
                rx={8}
                className="fill-[var(--surface-raised)] stroke-[var(--border)]"
                strokeWidth={1}
              />
              <foreignObject
                x={nodeX(i)}
                y={centerY - nodeHeight / 2}
                width={nodeWidth}
                height={nodeHeight}
              >
                <div className="flex h-full w-full items-center justify-center px-2 text-center text-xs font-medium text-[var(--foreground)]">
                  {node}
                </div>
              </foreignObject>
            </g>
          ))}
        </svg>
      </div>

      <p className="text-xs text-[var(--muted)]">
        Green edges cite a tool-call observation as evidence; amber edges do
        not (rejected by the guardrails-ai schema check before reaching this
        view in a correctly configured deployment).
      </p>
    </div>
  );
}
