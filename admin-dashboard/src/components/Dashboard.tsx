/**
 * SIE-X Admin Dashboard - Main Components
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  useTheme,
  Tab,
  Tabs,
  IconButton,
  Tooltip,
  Badge,
  Avatar,
  Chip
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { 
  TrendingUp,
  TrendingDown,
  Speed,
  Memory,
  Storage,
  CloudQueue,
  Security,
  Assessment,
  Timeline,
  AccountTree
} from '@mui/icons-material';
import { useQuery, useSubscription } from '@apollo/client';
import { formatDistance, formatDistanceToNow } from 'date-fns';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color?: string;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  icon,
  color = 'primary'
}) => {
  const theme = useTheme();
  const isPositive = change && change > 0;
  
  return (
    <Card elevation={2}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography variant="h4" component="h2">
              {typeof value === 'number' ? value.toLocaleString() : value}
            </Typography>
            {change !== undefined && (
              <Box display="flex" alignItems="center" mt={1}>
                {isPositive ? (
                  <TrendingUp style={{ color: theme.palette.success.main }} />
                ) : (
                  <TrendingDown style={{ color: theme.palette.error.main }} />
                )}
                <Typography
                  variant="body2"
                  style={{
                    color: isPositive 
                      ? theme.palette.success.main 
                      : theme.palette.error.main
                  }}
                >
                  {Math.abs(change)}%
                </Typography>
              </Box>
            )}
          </Box>
          <Avatar
            style={{
              backgroundColor: theme.palette[color].light,
              color: theme.palette[color].main
            }}
          >
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );
};

interface SystemHealthProps {
  data: {
    cpu: number;
    memory: number;
    gpu: number;
    disk: number;
  };
}

const SystemHealth: React.FC<SystemHealthProps> = ({ data }) => {
  const theme = useTheme();
  
  const getHealthColor = (value: number) => {
    if (value < 60) return theme.palette.success.main;
    if (value < 80) return theme.palette.warning.main;
    return theme.palette.error.main;
  };
  
  const healthMetrics = [
    { name: 'CPU', value: data.cpu, icon: <Speed /> },
    { name: 'Memory', value: data.memory, icon: <Memory /> },
    { name: 'GPU', value: data.gpu, icon: <CloudQueue /> },
    { name: 'Disk', value: data.disk, icon: <Storage /> }
  ];
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Grid container spacing={2}>
          {healthMetrics.map((metric) => (
            <Grid item xs={3} key={metric.name}>
              <Box textAlign="center">
                <Box
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  mb={1}
                >
                  {metric.icon}
                  <Typography variant="body2" ml={1}>
                    {metric.name}
                  </Typography>
                </Box>
                <Box position="relative" display="inline-flex">
                  <CircularProgress
                    variant="determinate"
                    value={metric.value}
                    size={60}
                    thickness={6}
                    style={{ color: getHealthColor(metric.value) }}
                  />
                  <Box
                    top={0}
                    left={0}
                    bottom={0}
                    right={0}
                    position="absolute"
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                  >
                    <Typography variant="caption" component="div">
                      {`${Math.round(metric.value)}%`}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

interface LiveExtractionMonitorProps {
  websocketUrl: string;
}

const LiveExtractionMonitor: React.FC<LiveExtractionMonitorProps> = ({
  websocketUrl
}) => {
  const [extractions, setExtractions] = useState<any[]>([]);
  const [stats, setStats] = useState({
    total: 0,
    successful: 0,
    failed: 0,
    avgDuration: 0
  });
  
  useEffect(() => {
    const ws = new WebSocket(websocketUrl);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'extraction') {
        setExtractions(prev => [data, ...prev].slice(0, 50));
        
        // Update stats
        setStats(prev => ({
          total: prev.total + 1,
          successful: data.status === 'success' ? prev.successful + 1 : prev.successful,
          failed: data.status === 'failed' ? prev.failed + 1 : prev.failed,
          avgDuration: (prev.avgDuration * prev.total + data.duration) / (prev.total + 1)
        }));
      }
    };
    
    return () => ws.close();
  }, [websocketUrl]);
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Live Extractions
        </Typography>
        
        <Grid container spacing={2} mb={2}>
          <Grid item xs={3}>
            <Typography variant="body2" color="textSecondary">
              Total
            </Typography>
            <Typography variant="h5">
              {stats.total}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="textSecondary">
              Success Rate
            </Typography>
            <Typography variant="h5" color="success.main">
              {stats.total > 0 
                ? `${((stats.successful / stats.total) * 100).toFixed(1)}%`
                : '0%'
              }
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="textSecondary">
              Failed
            </Typography>
            <Typography variant="h5" color="error.main">
              {stats.failed}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="textSecondary">
              Avg Duration
            </Typography>
            <Typography variant="h5">
              {stats.avgDuration.toFixed(0)}ms
            </Typography>
          </Grid>
        </Grid>
        
        <Box maxHeight={300} overflow="auto">
          {extractions.map((extraction, index) => (
            <Box
              key={extraction.id}
              p={1}
              mb={1}
              bgcolor="background.default"
              borderRadius={1}
              display="flex"
              alignItems="center"
              justifyContent="space-between"
            >
              <Box>
                <Typography variant="body2">
                  {extraction.user} - {extraction.mode}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {formatDistanceToNow(new Date(extraction.timestamp), {
                    addSuffix: true
                  })}
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={1}>
                <Chip
                  label={`${extraction.keywords} keywords`}
                  size="small"
                  variant="outlined"
                />
                <Chip
                  label={`${extraction.duration}ms`}
                  size="small"
                  color={extraction.duration < 500 ? 'success' : 'warning'}
                />
                <Chip
                  label={extraction.status}
                  size="small"
                  color={extraction.status === 'success' ? 'success' : 'error'}
                />
              </Box>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

interface ModelPerformanceProps {
  data: any[];
}

const ModelPerformance: React.FC<ModelPerformanceProps> = ({ data }) => {
  const theme = useTheme();
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Model Performance
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <RechartsTooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke={theme.palette.primary.main}
              name="Accuracy"
            />
            <Line
              type="monotone"
              dataKey="f1Score"
              stroke={theme.palette.secondary.main}
              name="F1 Score"
            />
            <Line
              type="monotone"
              dataKey="latency"
              stroke={theme.palette.warning.main}
              name="Latency (ms)"
              yAxisId="right"
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const [activeTab, setActiveTab] = useState(0);
  
  // GraphQL subscriptions for real-time data
  const { data: metricsData } = useSubscription(METRICS_SUBSCRIPTION);
  const { data: systemData } = useQuery(SYSTEM_HEALTH_QUERY, {
    pollInterval: 5000
  });
  
  const metrics = useMemo(() => ({
    totalExtractions: metricsData?.metrics.totalExtractions || 0,
    activeUsers: metricsData?.metrics.activeUsers || 0,
    avgResponseTime: metricsData?.metrics.avgResponseTime || 0,
    cacheHitRate: metricsData?.metrics.cacheHitRate || 0
  }), [metricsData]);
  
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        SIE-X Admin Dashboard
      </Typography>
      
      <Tabs
        value={activeTab}
        onChange={(e, v) => setActiveTab(v)}
        sx={{ mb: 3 }}
      >
        <Tab label="Overview" />
        <Tab label="Performance" />
        <Tab label="Users" />
        <Tab label="Audit" />
        <Tab label="Settings" />
      </Tabs>
      
      {activeTab === 0 && (
        <Grid container spacing={3}>
          {/* Metrics Row */}
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Total Extractions"
              value={metrics.totalExtractions}
              change={12.5}
              icon={<Assessment />}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Active Users"
              value={metrics.activeUsers}
              change={-5.2}
              icon={<AccountTree />}
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Avg Response Time"
              value={`${metrics.avgResponseTime}ms`}
              change={-8.1}
              icon={<Timeline />}
              color="secondary"
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard
              title="Cache Hit Rate"
              value={`${metrics.cacheHitRate}%`}
              change={3.7}
              icon={<Memory />}
              color="success"
            />
          </Grid>
          
          {/* System Health */}
          <Grid item xs={12} md={6}>
            <SystemHealth 
              data={systemData?.systemHealth || {
                cpu: 0,
                memory: 0,
                gpu: 0,
                disk: 0
              }}
            />
          </Grid>
          
          {/* Live Monitor */}
          <Grid item xs={12} md={6}>
            <LiveExtractionMonitor
              websocketUrl="wss://api.sie-x.com/admin/live"
            />
          </Grid>
          
          {/* Model Performance */}
          <Grid item xs={12}>
            <ModelPerformance
              data={metricsData?.modelPerformance || []}
            />
          </Grid>
        </Grid>
      )}
      
      {activeTab === 1 && <PerformanceTab />}
      {activeTab === 2 && <UsersTab />}
      {activeTab === 3 && <AuditTab />}
      {activeTab === 4 && <SettingsTab />}
    </Box>
  );
};

export default Dashboard;