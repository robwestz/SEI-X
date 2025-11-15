/**
 * SIE-X Node.js/TypeScript SDK
 * Enterprise-grade SDK for Semantic Intelligence Engine X
 */

import axios, { AxiosInstance } from 'axios';
import WebSocket from 'ws';
import { EventEmitter } from 'events';
import * as jwt from 'jsonwebtoken';
import { createHash } from 'crypto';
import pRetry from 'p-retry';
import PQueue from 'p-queue';

// Types
export enum ExtractionMode {
  FAST = 'fast',
  BALANCED = 'balanced',
  ADVANCED = 'advanced',
  ULTRA = 'ultra'
}

export enum OutputFormat {
  OBJECT = 'object',
  STRING = 'string',
  JSON = 'json'
}

export interface Keyword {
  text: string;
  score: number;
  type: string;
  count: number;
  relatedTerms: string[];
  confidence: number;
  semanticCluster?: number;
}

export interface ExtractionOptions {
  topK?: number;
  mode?: ExtractionMode;
  enableClustering?: boolean;
  minConfidence?: number;
  language?: string;
  outputFormat?: OutputFormat;
}

export interface BatchOptions extends ExtractionOptions {
  batchSize?: number;
  concurrency?: number;
}

export interface AuthConfig {
  apiKey?: string;
  jwtToken?: string;
  oauthClientId?: string;
  oauthClientSecret?: string;
}

export interface ClientConfig {
  baseUrl?: string;
  auth?: AuthConfig;
  timeout?: number;
  maxRetries?: number;
  enableCaching?: boolean;
}

// Authentication Handler
class AuthHandler {
  private config: AuthConfig;
  private tokenExpiry?: Date;

  constructor(config: AuthConfig) {
    this.config = config;
  }

  async getHeaders(): Promise<Record<string, string>> {
    if (this.config.apiKey) {
      return { Authorization: `ApiKey ${this.config.apiKey}` };
    } else if (this.config.jwtToken) {
      if (this.isTokenExpired()) {
        await this.refreshToken();
      }
      return { Authorization: `Bearer ${this.config.jwtToken}` };
    } else if (this.config.oauthClientId) {
      if (!this.config.jwtToken || this.isTokenExpired()) {
        await this.oauthAuthenticate();
      }
      return { Authorization: `Bearer ${this.config.jwtToken}` };
    }
    throw new Error('No authentication method configured');
  }

  private isTokenExpired(): boolean {
    if (!this.tokenExpiry) return true;
    return new Date() >= this.tokenExpiry;
  }

  private async refreshToken(): Promise<void> {
    // Implement token refresh logic
  }

  private async oauthAuthenticate(): Promise<void> {
    // Implement OAuth2 authentication
  }
}

// Main Client
export class SIEXClient {
  private baseUrl: string;
  private auth: AuthHandler;
  private axios: AxiosInstance;
  private cache: Map<string, any>;
  private enableCaching: boolean;
  private maxRetries: number;

  constructor(config: ClientConfig = {}) {
    this.baseUrl = config.baseUrl || 'https://api.sie-x.com';
    this.auth = new AuthHandler(config.auth || {});
    this.enableCaching = config.enableCaching ?? true;
    this.maxRetries = config.maxRetries || 3;
    this.cache = new Map();

    this.axios = axios.create({
      baseURL: this.baseUrl,
      timeout: config.timeout || 30000,
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor for auth
    this.axios.interceptors.request.use(async (config) => {
      const headers = await this.auth.getHeaders();
      config.headers = { ...config.headers, ...headers };
      return config;
    });

    // Response interceptor for error handling
    this.axios.interceptors.response.use(
      response => response,
      async error => {
        if (error.response?.status === 429) {
          // Rate limit handling
          const retryAfter = error.response.headers['retry-after'] || 1;
          await this.sleep(retryAfter * 1000);
          return this.axios.request(error.config);
        }
        throw error;
      }
    );
  }

  /**
   * Extract keywords from text
   * @example
   * ```typescript
   * const client = new SIEXClient({ auth: { apiKey: 'your-key' } });
   * const keywords = await client.extract('Your text here', {
   *   topK: 20,
   *   mode: ExtractionMode.ADVANCED
   * });
   * ```
   */
  async extract(
    text: string | string[],
    options: ExtractionOptions = {}
  ): Promise<Keyword[] | string[] | any> {
    // Check cache for single text
    if (this.enableCaching && typeof text === 'string') {
      const cacheKey = this.getCacheKey(text, options);
      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey);
      }
    }

    const payload = {
      text,
      top_k: options.topK || 10,
      mode: options.mode || ExtractionMode.BALANCED,
      enable_clustering: options.enableClustering ?? true,
      min_confidence: options.minConfidence || 0.3,
      ...(options.language && { language: options.language })
    };

    const response = await pRetry(
      () => this.axios.post('/extract', payload),
      { retries: this.maxRetries }
    );

    let result: any;
    
    switch (options.outputFormat) {
      case OutputFormat.STRING:
        result = response.data.keywords.map((kw: any) => kw.text);
        break;
      case OutputFormat.JSON:
        result = response.data;
        break;
      default:
        result = response.data.keywords.map((kw: any) => ({
          text: kw.text,
          score: kw.score,
          type: kw.type,
          count: kw.count,
          relatedTerms: kw.related_terms,
          confidence: kw.confidence,
          semanticCluster: kw.semantic_cluster
        }));
    }

    // Cache result
    if (this.enableCaching && typeof text === 'string') {
      const cacheKey = this.getCacheKey(text, options);
      this.cache.set(cacheKey, result);
    }

    return result;
  }

  /**
   * Process multiple documents in batch
   */
  async extractBatch(
    documents: string[],
    options: BatchOptions = {}
  ): Promise<Keyword[][]> {
    const queue = new PQueue({ 
      concurrency: options.concurrency || 5 
    });
    
    const batchSize = options.batchSize || 50;
    const results: Keyword[][] = [];

    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      queue.add(async () => {
        const batchResults = await this.extract(batch, options);
        results.push(...batchResults);
      });
    }

    await queue.onIdle();
    return results;
  }

  /**
   * Stream extraction results via WebSocket
   */
  streamExtract(
    text: string,
    options: ExtractionOptions = {}
  ): EventEmitter {
    const emitter = new EventEmitter();
    
    (async () => {
      try {
        const headers = await this.auth.getHeaders();
        const wsUrl = this.baseUrl
          .replace('https://', 'wss://')
          .replace('http://', 'ws://') + '/extract/stream';

        const ws = new WebSocket(wsUrl, { headers });

        ws.on('open', () => {
          ws.send(JSON.stringify({
            type: 'extract',
            text,
            options
          }));
        });

        ws.on('message', (data: string) => {
          const message = JSON.parse(data);
          
          if (message.type === 'complete') {
            emitter.emit('complete');
            ws.close();
          } else {
            emitter.emit('chunk', message);
          }
        });

        ws.on('error', (error) => {
          emitter.emit('error', error);
        });

        ws.on('close', () => {
          emitter.emit('close');
        });

      } catch (error) {
        emitter.emit('error', error);
      }
    })();

    return emitter;
  }

  /**
   * Analyze relationships across multiple documents
   */
  async analyzeMultiple(
    documents: string[],
    topKCommon: number = 10,
    topKDistinctive: number = 5
  ): Promise<any> {
    const response = await this.axios.post('/analyze/multi', {
      documents,
      options: {
        top_k_common: topKCommon,
        top_k_distinctive: topKDistinctive
      }
    });

    return response.data;
  }

  /**
   * Get available models
   */
  async getModels(): Promise<any[]> {
    const response = await this.axios.get('/models');
    return response.data;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<any> {
    const response = await this.axios.get('/health');
    return response.data;
  }

  private getCacheKey(text: string, options: ExtractionOptions): string {
    const keyData = `${text}:${options.topK}:${options.mode}:${options.minConfidence}`;
    return createHash('sha256').update(keyData).digest('hex');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Convenience function
export async function extractKeywords(
  text: string,
  apiKey: string,
  options?: ExtractionOptions
): Promise<Keyword[]> {
  const client = new SIEXClient({ auth: { apiKey } });
  return await client.extract(text, options) as Keyword[];
}