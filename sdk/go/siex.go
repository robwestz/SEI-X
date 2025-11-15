// Package siex provides a Go SDK for the Semantic Intelligence Engine X
package siex

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/gorilla/websocket"
	"golang.org/x/sync/semaphore"
)

// ExtractionMode defines the extraction mode
type ExtractionMode string

const (
	ModeFast     ExtractionMode = "fast"
	ModeBalanced ExtractionMode = "balanced"
	ModeAdvanced ExtractionMode = "advanced"
	ModeUltra    ExtractionMode = "ultra"
)

// OutputFormat defines the output format
type OutputFormat string

const (
	FormatObject OutputFormat = "object"
	FormatString OutputFormat = "string"
	FormatJSON   OutputFormat = "json"
)

// Keyword represents an extracted keyword
type Keyword struct {
	Text            string   `json:"text"`
	Score           float64  `json:"score"`
	Type            string   `json:"type"`
	Count           int      `json:"count"`
	RelatedTerms    []string `json:"related_terms"`
	Confidence      float64  `json:"confidence"`
	SemanticCluster *int     `json:"semantic_cluster,omitempty"`
}

// ExtractionOptions configures extraction behavior
type ExtractionOptions struct {
	TopK             int            `json:"top_k"`
	Mode             ExtractionMode `json:"mode"`
	EnableClustering bool           `json:"enable_clustering"`
	MinConfidence    float64        `json:"min_confidence"`
	Language         string         `json:"language,omitempty"`
	OutputFormat     OutputFormat   `json:"output_format"`
}

// DefaultOptions returns default extraction options
func DefaultOptions() *ExtractionOptions {
	return &ExtractionOptions{
		TopK:             10,
		Mode:             ModeBalanced,
		EnableClustering: true,
		MinConfidence:    0.3,
		OutputFormat:     FormatObject,
	}
}

// AuthConfig configures authentication
type AuthConfig struct {
	APIKey            string
	JWTToken          string
	OAuthClientID     string
	OAuthClientSecret string
}

// ClientConfig configures the client
type ClientConfig struct {
	BaseURL       string
	Auth          *AuthConfig
	Timeout       time.Duration
	MaxRetries    int
	EnableCaching bool
	Concurrency   int
}

// DefaultConfig returns default client configuration
func DefaultConfig() *ClientConfig {
	return &ClientConfig{
		BaseURL:       "https://api.sie-x.com",
		Timeout:       30 * time.Second,
		MaxRetries:    3,
		EnableCaching: true,
		Concurrency:   5,
	}
}

// Client is the main SIE-X client
type Client struct {
	config     *ClientConfig
	httpClient *http.Client
	cache      sync.Map
	sem        *semaphore.Weighted
}

// NewClient creates a new SIE-X client
func NewClient(config *ClientConfig) *Client {
	if config == nil {
		config = DefaultConfig()
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		sem: semaphore.NewWeighted(int64(config.Concurrency)),
	}
}

// Extract extracts keywords from text
func (c *Client) Extract(ctx context.Context, text interface{}, opts *ExtractionOptions) (interface{}, error) {
	if opts == nil {
		opts = DefaultOptions()
	}

	// Check cache for single string
	if textStr, ok := text.(string); ok && c.config.EnableCaching {
		cacheKey := c.getCacheKey(textStr, opts)
		if cached, found := c.cache.Load(cacheKey); found {
			return cached, nil
		}
	}

	// Prepare request
	payload := map[string]interface{}{
		"text":              text,
		"top_k":             opts.TopK,
		"mode":              opts.Mode,
		"enable_clustering": opts.EnableClustering,
		"min_confidence":    opts.MinConfidence,
	}

	if opts.Language != "" {
		payload["language"] = opts.Language
	}

	// Make request with retry
	var result interface{}
	err := backoff.Retry(func() error {
		resp, err := c.request(ctx, "POST", "/extract", payload)
		if err != nil {
			return err
		}

		result = c.parseResponse(resp, opts.OutputFormat)
		return nil
	}, backoff.WithMaxRetries(backoff.NewExponentialBackOff(), uint64(c.config.MaxRetries)))

	if err != nil {
		return nil, err
	}

	// Cache result
	if textStr, ok := text.(string); ok && c.config.EnableCaching {
		cacheKey := c.getCacheKey(textStr, opts)
		c.cache.Store(cacheKey, result)
	}

	return result, nil
}

// ExtractBatch processes multiple documents
func (c *Client) ExtractBatch(ctx context.Context, documents []string, opts *ExtractionOptions) ([][]Keyword, error) {
	results := make([][]Keyword, len(documents))
	var wg sync.WaitGroup
	errors := make(chan error, len(documents))

	for i, doc := range documents {
		wg.Add(1)
		go func(idx int, document string) {
			defer wg.Done()

			// Acquire semaphore
			if err := c.sem.Acquire(ctx, 1); err != nil {
				errors <- err
				return
			}
			defer c.sem.Release(1)

			// Extract keywords
			keywords, err := c.Extract(ctx, document, opts)
			if err != nil {
				errors <- err
				return
			}

			if kws, ok := keywords.([]Keyword); ok {
				results[idx] = kws
			}
		}(i, doc)
	}

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// StreamExtract streams extraction results via WebSocket
func (c *Client) StreamExtract(ctx context.Context, text string, opts *ExtractionOptions) (<-chan StreamResult, error) {
	results := make(chan StreamResult)

	wsURL := c.config.BaseURL
	wsURL = replaceProtocol(wsURL, "wss://", "ws://")
	wsURL += "/extract/stream"

	headers := c.getAuthHeaders()
	dialer := websocket.Dialer{
		HandshakeTimeout: c.config.Timeout,
	}

	conn, _, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		return nil, err
	}

	// Send extraction request
	request := map[string]interface{}{
		"type":    "extract",
		"text":    text,
		"options": opts,
	}

	if err := conn.WriteJSON(request); err != nil {
		conn.Close()
		return nil, err
	}

	go func() {
		defer close(results)
		defer conn.Close()

		for {
			var msg StreamResult
			if err := conn.ReadJSON(&msg); err != nil {
				if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway) {
					results <- StreamResult{Error: err}
				}
				return
			}

			results <- msg

			if msg.Type == "complete" {
				return
			}
		}
	}()

	return results, nil
}

// AnalyzeMultiple analyzes relationships across documents
func (c *Client) AnalyzeMultiple(ctx context.Context, documents []string, topKCommon, topKDistinctive int) (*MultiDocAnalysis, error) {
	payload := map[string]interface{}{
		"documents": documents,
		"options": map[string]int{
			"top_k_common":      topKCommon,
			"top_k_distinctive": topKDistinctive,
		},
	}

	resp, err := c.request(ctx, "POST", "/analyze/multi", payload)
	if err != nil {
		return nil, err
	}

	var analysis MultiDocAnalysis
	if err := json.Unmarshal(resp, &analysis); err != nil {
		return nil, err
	}

	return &analysis, nil
}

// request makes an HTTP request
func (c *Client) request(ctx context.Context, method, endpoint string, payload interface{}) ([]byte, error) {
	url := c.config.BaseURL + endpoint

	var body io.Reader
	if payload != nil {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		body = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, err
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.getAuthHeaders() {
		req.Header.Set(k, v)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// Helper functions
func (c *Client) getAuthHeaders() map[string]string {
	headers := make(map[string]string)

	if c.config.Auth != nil {
		if c.config.Auth.APIKey != "" {
			headers["Authorization"] = "ApiKey " + c.config.Auth.APIKey
		} else if c.config.Auth.JWTToken != "" {
			headers["Authorization"] = "Bearer " + c.config.Auth.JWTToken
		}
	}

	return headers
}

func (c *Client) getCacheKey(text string, opts *ExtractionOptions) string {
	keyData := fmt.Sprintf("%s:%d:%s:%.2f", text, opts.TopK, opts.Mode, opts.MinConfidence)
	hash := sha256.Sum256([]byte(keyData))
	return fmt.Sprintf("%x", hash)
}

func (c *Client) parseResponse(data []byte, format OutputFormat) interface{} {
	var resp map[string]interface{}
	json.Unmarshal(data, &resp)

	switch format {
	case FormatString:
		// Extract text fields
		keywords := resp["keywords"].([]interface{})
		texts := make([]string, len(keywords))
		for i, kw := range keywords {
			texts[i] = kw.(map[string]interface{})["text"].(string)
		}
		return texts
	case FormatJSON:
		return resp
	default:
		// Parse as Keyword objects
		var result struct {
			Keywords []Keyword `json:"keywords"`
		}
		json.Unmarshal(data, &result)
		return result.Keywords
	}
}

// StreamResult represents a streaming result
type StreamResult struct {
	Type      string    `json:"type"`
	ChunkIndex int      `json:"chunk_index"`
	Keywords  []Keyword `json:"keywords"`
	Error     error     `json:"-"`
}

// MultiDocAnalysis represents multi-document analysis results
type MultiDocAnalysis struct {
	CommonKeywords     []Keyword   `json:"common_keywords"`
	DistinctivePerDoc  [][]Keyword `json:"distinctive_per_doc"`
	DocumentClusters   []int       `json:"document_clusters"`
	Statistics         map[string]interface{} `json:"statistics"`
}

func replaceProtocol(url, httpsReplace, httpReplace string) string {
	if len(url) >= 8 && url[:8] == "https://" {
		return httpsReplace + url[8:]
	} else if len(url) >= 7 && url[:7] == "http://" {
		return httpReplace + url[7:]
	}
	return url
}