"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertCircle, TrendingDown, Shield } from "lucide-react";

interface RobustnessData {
  clean_run_id: string;
  adv_run_id: string;
  clean_run_name: string;
  adv_run_name: string;
  overall_robustness: {
    delta_map: number;
    delta_map50: number;
    drop_percentage: number;
    robustness_ratio: number;
    delta_recall: number;
    delta_precision: number;
    ap_clean: number;
    ap_adv: number;
    ap50_clean: number;
    ap50_adv: number;
    recall_clean: number;
    recall_adv: number;
    precision_clean: number;
    precision_adv: number;
  };
  per_class_robustness: Record<string, any>;
  attack_info?: {
    id: string;
    name: string;
    attack_type: string;
    parameters?: Record<string, any>;
    target_class?: string;
  };
  visualization_data: {
    overall_comparison: Array<{
      metric: string;
      clean: number;
      adversarial: number;
    }>;
    drops: Array<{
      metric: string;
      drop_percentage: number;
      delta: number;
    }>;
    per_class_comparison: Array<{
      class_name: string;
      clean_map: number;
      adv_map: number;
      drop_percentage: number;
      robustness_ratio: number;
    }>;
    summary: {
      max_drop: {
        metric: string;
        value: number;
        severity: string;
      };
      most_vulnerable_class: {
        class: string;
        drop: number;
      } | null;
      overall_robustness_ratio: number;
    };
  };
}

interface RobustnessComparisonProps {
  data: RobustnessData;
}

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "critical":
      return "destructive";
    case "high":
      return "destructive";
    case "medium":
      return "default";
    default:
      return "secondary";
  }
};

const getSeverityBadge = (severity: string) => {
  const colorMap: Record<string, string> = {
    critical: "bg-red-500",
    high: "bg-orange-500",
    medium: "bg-yellow-500",
    low: "bg-green-500",
  };
  return colorMap[severity] || colorMap.medium;
};

export function RobustnessComparison({ data }: RobustnessComparisonProps) {
  const { visualization_data, overall_robustness, attack_info } = data;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* mAP Drop Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">mAP Drop</CardTitle>
            <TrendingDown className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {visualization_data.summary.max_drop.value.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {overall_robustness.ap_clean.toFixed(3)} → {overall_robustness.ap_adv.toFixed(3)}
            </p>
            <Badge className={`mt-2 ${getSeverityBadge(visualization_data.summary.max_drop.severity)}`}>
              {visualization_data.summary.max_drop.severity.toUpperCase()}
            </Badge>
          </CardContent>
        </Card>

        {/* Robustness Ratio Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Robustness Ratio</CardTitle>
            <Shield className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(visualization_data.summary.overall_robustness_ratio * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Model retained {(visualization_data.summary.overall_robustness_ratio * 100).toFixed(1)}% performance
            </p>
            <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full"
                style={{ width: `${visualization_data.summary.overall_robustness_ratio * 100}%` }}
              />
            </div>
          </CardContent>
        </Card>

        {/* Most Vulnerable Class Card */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Most Vulnerable</CardTitle>
            <AlertCircle className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {visualization_data.summary.most_vulnerable_class?.class || "N/A"}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {visualization_data.summary.most_vulnerable_class?.drop.toFixed(1)}% drop
            </p>
            <Badge variant="destructive" className="mt-2">
              CRITICAL
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Attack Information */}
      {attack_info && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Attack Information</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Attack Type:</span> {attack_info.attack_type}
              </div>
              <div>
                <span className="font-medium">Attack Name:</span> {attack_info.name}
              </div>
              {attack_info.parameters && (
                <div className="col-span-2">
                  <span className="font-medium">Parameters:</span>{" "}
                  <code className="text-xs bg-gray-100 px-2 py-1 rounded">
                    {JSON.stringify(attack_info.parameters)}
                  </code>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Overall Performance Comparison Bar Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Overall Performance Comparison</CardTitle>
          <CardDescription>Clean vs Adversarial performance across metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={visualization_data.overall_comparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 1]} tickFormatter={(value) => value.toFixed(2)} />
              <Tooltip formatter={(value: any) => value.toFixed(3)} />
              <Legend />
              <Bar dataKey="clean" fill="#10b981" name="Clean" />
              <Bar dataKey="adversarial" fill="#ef4444" name="Adversarial" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Performance Drop Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Drop by Metric</CardTitle>
          <CardDescription>Percentage drop for each metric</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={visualization_data.drops} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} unit="%" />
              <YAxis type="category" dataKey="metric" width={100} />
              <Tooltip formatter={(value: any) => `${value.toFixed(1)}%`} />
              <Bar dataKey="drop_percentage" fill="#f59e0b" name="Drop %">
                {visualization_data.drops.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.drop_percentage > 70
                        ? "#ef4444"
                        : entry.drop_percentage > 50
                        ? "#f59e0b"
                        : "#10b981"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

    </div>
  );
}
