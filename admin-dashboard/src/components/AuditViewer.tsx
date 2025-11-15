/**
 * Audit Log and Data Lineage Viewer
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  IconButton,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  DateTimePicker
} from '@mui/material';
import {
  Search,
  FilterList,
  GetApp,
  Visibility,
  AccountTree,
  Timeline
} from '@mui/icons-material';
import { DataGrid, GridColDef } from '@mui/x-data-grid';
import * as d3 from 'd3';
import { format } from 'date-fns';

interface AuditLog {
  id: string;
  timestamp: string;
  eventType: string;
  userId: string;
  resourceId: string;
  resourceType: string;
  action: string;
  status: string;
  durationMs: number;
  metadata: any;
}

interface LineageNode {
  id: string;
  type: string;
  operation: string;
  timestamp: string;
  metadata: any;
}

interface LineageVisualizerProps {
  nodeId: string;
  data: {
    nodes: LineageNode[];
    edges: Array<{ from: string; to: string }>;
  };
}

const LineageVisualizer: React.FC<LineageVisualizerProps> = ({ nodeId, data }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  useEffect(() => {
    if (!svgRef.current || !data) return;
    
    const width = 800;
    const height = 600;
    
    // Clear previous graph
    d3.select(svgRef.current).selectAll('*').remove();
    
    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);
    
    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.edges).id((d: any) => d.id))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));
    
    // Add links
    const link = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .enter().append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', 2);
    
    // Add nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter().append('circle')
      .attr('r', 8)
      .attr('fill', (d: any) => {
        if (d.id === nodeId) return '#ff4081';
        if (d.type === 'upstream') return '#3f51b5';
        return '#4caf50';
      })
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter().append('text')
      .text((d: any) => d.label)
      .attr('font-size', 10)
      .attr('dx', 12)
      .attr('dy', 4);
    
    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);
      
      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);
      
      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });
    
    function dragstarted(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event: any, d: any) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event: any, d: any) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }, [nodeId, data]);
  
  return (
    <Box>
      <svg ref={svgRef} style={{ width: '100%', height: '600px' }} />
    </Box>
  );
};

const AuditViewer: React.FC = () => {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [filters, setFilters] = useState({
    eventType: '',
    userId: '',
    status: '',
    startDate: null,
    endDate: null
  });
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);
  const [lineageData, setLineageData] = useState<any>(null);
  
  const columns: GridColDef[] = [
    {
      field: 'timestamp',
      headerName: 'Timestamp',
      width: 180,
      renderCell: (params) => format(new Date(params.value), 'yyyy-MM-dd HH:mm:ss')
    },
    {
      field: 'eventType',
      headerName: 'Event Type',
      width: 150,
      renderCell: (params) => (
        <Chip label={params.value} size="small" variant="outlined" />
      )
    },
    {
      field: 'userId',
      headerName: 'User',
      width: 150
    },
    {
      field: 'resourceType',
      headerName: 'Resource Type',
      width: 120
    },
    {
      field: 'action',
      headerName: 'Action',
      width: 100
    },
    {
      field: 'status',
      headerName: 'Status',
      width: 100,
      renderCell: (params) => (
        <Chip
          label={params.value}
          size="small"
          color={params.value === 'success' ? 'success' : 'error'}
        />
      )
    },
    {
      field: 'durationMs',
      headerName: 'Duration',
      width: 100,
      renderCell: (params) => `${params.value}ms`
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      sortable: false,
      renderCell: (params) => (
        <>
          <IconButton
            size="small"
            onClick={() => handleViewDetails(params.row)}
          >
            <Visibility />
          </IconButton>
          <IconButton
            size="small"
            onClick={() => handleViewLineage(params.row)}
          >
            <AccountTree />
          </IconButton>
        </>
      )
    }
  ];
  
  const handleViewDetails = (log: AuditLog) => {
    setSelectedLog(log);
  };
  
  const handleViewLineage = async (log: AuditLog) => {
    // Fetch lineage data
    const response = await fetch(`/api/lineage/${log.resourceId}`);
    const data = await response.json();
    setLineageData(data);
  };
  
  const handleExport = () => {
    // Export filtered logs
    const csvContent = [
      Object.keys(logs[0]).join(','),
      ...logs.map(log => Object.values(log).join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit-logs-${Date.now()}.csv`;
    a.click();
  };
  
  return (
    <Box>
      {/* Filters */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Event Type</InputLabel>
              <Select
                value={filters.eventType}
                onChange={(e) => setFilters({...filters, eventType: e.target.value})}
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="extraction_started">Extraction Started</MenuItem>
                <MenuItem value="extraction_completed">Extraction Completed</MenuItem>
                <MenuItem value="extraction_failed">Extraction Failed</MenuItem>
                <MenuItem value="model_updated">Model Updated</MenuItem>
                <MenuItem value="security_event">Security Event</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <TextField
              fullWidth
              size="small"
              label="User ID"
              value={filters.userId}
              onChange={(e) => setFilters({...filters, userId: e.target.value})}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Status</InputLabel>
              <Select
                value={filters.status}
                onChange={(e) => setFilters({...filters, status: e.target.value})}
              >
                <MenuItem value="">All</MenuItem>
                <MenuItem value="success">Success</MenuItem>
                <MenuItem value="failed">Failed</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <DateTimePicker
              label="Start Date"
              value={filters.startDate}
              onChange={(date) => setFilters({...filters, startDate: date})}
              renderInput={(params) => <TextField {...params} size="small" fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <DateTimePicker
              label="End Date"
              value={filters.endDate}
              onChange={(date) => setFilters({...filters, endDate: date})}
              renderInput={(params) => <TextField {...params} size="small" fullWidth />}
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box display="flex" gap={1}>
              <Button
                variant="contained"
                startIcon={<Search />}
                onClick={() => {/* Apply filters */}}
              >
                Search
              </Button>
              <IconButton onClick={handleExport}>
                <GetApp />
              </IconButton>
            </Box>
          </Grid>
        </Grid>
      </Paper>
      
      {/* Data Grid */}
      <Paper sx={{ height: 600, width: '100%' }}>
        <DataGrid
          rows={logs}
          columns={columns}
          pageSize={rowsPerPage}
          onPageSizeChange={(newPageSize) => setRowsPerPage(newPageSize)}
          rowsPerPageOptions={[25, 50, 100]}
          checkboxSelection
          disableSelectionOnClick
        />
      </Paper>
      
      {/* Details Dialog */}
      <Dialog
        open={!!selectedLog}
        onClose={() => setSelectedLog(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Audit Log Details</DialogTitle>
        <DialogContent>
          {selectedLog && (
            <Box>
              <Typography variant="body2" gutterBottom>
                <strong>ID:</strong> {selectedLog.id}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Timestamp:</strong> {format(new Date(selectedLog.timestamp), 'yyyy-MM-dd HH:mm:ss')}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Event Type:</strong> {selectedLog.eventType}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>User:</strong> {selectedLog.userId}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Resource:</strong> {selectedLog.resourceType} - {selectedLog.resourceId}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Action:</strong> {selectedLog.action}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Status:</strong> {selectedLog.status}
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Duration:</strong> {selectedLog.durationMs}ms
              </Typography>
              <Typography variant="body2" gutterBottom>
                <strong>Metadata:</strong>
              </Typography>
              <Paper variant="outlined" sx={{ p: 1, bgcolor: 'grey.50' }}>
                <pre style={{ margin: 0, fontSize: '12px' }}>
                  {JSON.stringify(selectedLog.metadata, null, 2)}
                </pre>
              </Paper>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSelectedLog(null)}>Close</Button>
        </DialogActions>
      </Dialog>
      
      {/* Lineage Dialog */}
      <Dialog
        open={!!lineageData}
        onClose={() => setLineageData(null)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Data Lineage
          <IconButton
            onClick={() => setLineageData(null)}
            style={{ position: 'absolute', right: 8, top: 8 }}
          >
            <Close />
          </IconButton>
        </DialogTitle>
        <DialogContent>
          {lineageData && (
            <LineageVisualizer
              nodeId={lineageData.nodeId}
              data={lineageData.graph}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default AuditViewer;