"""
TypeScript code generator for Universal Code Modernization Platform
Generates modern TypeScript code with type safety and best practices
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language_targets import LanguageGenerator, LanguageFeatures, CodeGenerationContext, MigrationStrategy
from typing import Dict, List, Any
import json
from datetime import datetime


class TypeScriptGenerator(LanguageGenerator):
    """Generator for TypeScript programming language"""
    
    def __init__(self):
        features = LanguageFeatures(
            name="TypeScript",
            version="5.3",
            has_generics=True,
            has_async_await=True,
            has_pattern_matching=False,
            has_null_safety=True,
            has_memory_safety=False,
            has_type_inference=True,
            has_traits_interfaces=True,
            concurrency_model="event_loop",
            error_handling="exceptions",
            package_manager="npm",
            package_file="package.json",
            build_system="tsc",
            build_file="tsconfig.json",
            web_frameworks=["express", "fastify", "nest", "koa"],
            testing_frameworks=["jest", "vitest", "mocha", "playwright"],
            orm_libraries=["prisma", "typeorm", "sequelize", "drizzle"]
        )
        super().__init__(features)
        
        self.type_mappings = {
            "string": "string",
            "str": "string",
            "int": "number",
            "long": "number",
            "float": "number",
            "double": "number",
            "boolean": "boolean",
            "bool": "boolean",
            "void": "void",
            "list": "Array",
            "array": "Array",
            "map": "Map",
            "dict": "Record",
        }
        
        self.common_imports = {
            "web": ["express", "@types/express", "cors", "@types/cors"],
            "cli": ["commander", "@types/node"],
            "library": ["@types/node"],
            "service": ["express", "helmet", "morgan", "compression"]
        }
    
    def generate_project_structure(self, context: CodeGenerationContext) -> Dict[str, str]:
        """Generate TypeScript project structure with package.json and config files"""
        
        project_name = self._sanitize_package_name(context.domain)
        dependencies = self._get_dependencies(context)
        dev_dependencies = self._get_dev_dependencies(context)
        
        package_json = {
            "name": project_name,
            "version": "0.1.0",
            "description": f"Modernized {context.source_language} to TypeScript migration",
            "main": "dist/index.js",
            "types": "dist/index.d.ts",
            "scripts": {
                "build": "tsc",
                "start": "node dist/index.js",
                "dev": "ts-node-dev --respawn --transpile-only src/index.ts",
                "test": "jest",
                "test:watch": "jest --watch",
                "test:coverage": "jest --coverage",
                "lint": "eslint src/**/*.ts",
                "lint:fix": "eslint src/**/*.ts --fix",
                "format": "prettier --write src/**/*.ts",
                "clean": "rimraf dist",
                "prebuild": "npm run clean",
                "postbuild": "npm run copy-assets",
                "copy-assets": "copyfiles -u 1 src/**/*.json src/**/*.yaml dist/",
                "typecheck": "tsc --noEmit"
            },
            "keywords": ["typescript", "modernization", "pomegrantemuse"],
            "author": "PomegranteMuse Generated",
            "license": "MIT",
            "dependencies": dependencies,
            "devDependencies": dev_dependencies,
            "engines": {
                "node": ">=18.0.0",
                "npm": ">=9.0.0"
            }
        }

        tsconfig_json = {
            "compilerOptions": {
                "target": "ES2022",
                "lib": ["ES2022", "DOM"],
                "module": "commonjs",
                "moduleResolution": "node",
                "declaration": True,
                "outDir": "./dist",
                "rootDir": "./src",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True,
                "resolveJsonModule": True,
                "allowSyntheticDefaultImports": True,
                "experimentalDecorators": True,
                "emitDecoratorMetadata": True,
                "sourceMap": True,
                "incremental": True,
                "noImplicitAny": True,
                "noImplicitReturns": True,
                "noImplicitThis": True,
                "noUnusedLocals": True,
                "noUnusedParameters": True,
                "exactOptionalPropertyTypes": True
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
        }

        jest_config = {
            "preset": "ts-jest",
            "testEnvironment": "node",
            "roots": ["<rootDir>/src"],
            "testMatch": ["**/__tests__/**/*.ts", "**/*.(test|spec).ts"],
            "transform": {
                "^.+\\.ts$": "ts-jest"
            },
            "collectCoverageFrom": [
                "src/**/*.ts",
                "!src/**/*.d.ts",
                "!src/**/*.test.ts",
                "!src/**/*.spec.ts"
            ],
            "coverageDirectory": "coverage",
            "coverageReporters": ["text", "lcov", "html"]
        }

        eslint_config = {
            "parser": "@typescript-eslint/parser",
            "extends": [
                "@typescript-eslint/recommended",
                "prettier"
            ],
            "plugins": ["@typescript-eslint"],
            "parserOptions": {
                "ecmaVersion": 2022,
                "sourceType": "module"
            },
            "rules": {
                "@typescript-eslint/no-unused-vars": "error",
                "@typescript-eslint/no-explicit-any": "warn",
                "@typescript-eslint/explicit-function-return-type": "warn",
                "prefer-const": "error",
                "no-var": "error"
            }
        }

        prettier_config = {
            "semi": True,
            "trailingComma": "es5",
            "singleQuote": True,
            "printWidth": 80,
            "tabWidth": 2,
            "useTabs": False
        }

        index_ts = '''/**
 * Main application entry point
 * Auto-generated TypeScript code from PomegranteMuse
 */

import { Application } from './core/Application';
import { Config } from './config/Config';
import { Logger } from './utils/Logger';

const logger = Logger.create('Main');

async function main(): Promise<void> {
  try {
    logger.info('Starting application...');
    
    // Load configuration
    const config = await Config.load();
    
    // Create and start application
    const app = new Application(config);
    await app.start();
    
    // Setup graceful shutdown
    process.on('SIGINT', () => shutdown(app));
    process.on('SIGTERM', () => shutdown(app));
    
    logger.info('Application started successfully');
  } catch (error) {
    logger.error('Failed to start application:', error);
    process.exit(1);
  }
}

async function shutdown(app: Application): Promise<void> {
  logger.info('Shutting down gracefully...');
  
  try {
    await app.stop();
    logger.info('Application stopped successfully');
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown:', error);
    process.exit(1);
  }
}

// Only run if this file is executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

export { main };
'''

        config_ts = '''/**
 * Configuration management
 * Handles loading and validation of application configuration
 */

export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl?: boolean;
}

export interface ServerConfig {
  port: number;
  host: string;
  cors: {
    enabled: boolean;
    origins: string[];
  };
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  format: 'json' | 'pretty';
  file?: string;
}

export interface AppConfig {
  app: {
    name: string;
    version: string;
    environment: 'development' | 'test' | 'production';
  };
  server: ServerConfig;
  database?: DatabaseConfig;
  logging: LoggingConfig;
  apiKeys: Record<string, string>;
}

export class Config {
  private constructor(private readonly config: AppConfig) {}

  public static async load(): Promise<Config> {
    const config: AppConfig = {
      app: {
        name: process.env.APP_NAME || 'PomegranteMuse Generated App',
        version: process.env.APP_VERSION || '0.1.0',
        environment: (process.env.NODE_ENV as AppConfig['app']['environment']) || 'development',
      },
      server: {
        port: parseInt(process.env.PORT || '3000', 10),
        host: process.env.HOST || '0.0.0.0',
        cors: {
          enabled: process.env.CORS_ENABLED === 'true',
          origins: process.env.CORS_ORIGINS?.split(',') || ['*'],
        },
      },
      logging: {
        level: (process.env.LOG_LEVEL as LoggingConfig['level']) || 'info',
        format: (process.env.LOG_FORMAT as LoggingConfig['format']) || 'pretty',
        file: process.env.LOG_FILE,
      },
      apiKeys: Config.parseApiKeys(process.env.API_KEYS),
    };

    // Add database config if DATABASE_URL is provided
    if (process.env.DATABASE_URL) {
      config.database = Config.parseDatabaseUrl(process.env.DATABASE_URL);
    }

    const configInstance = new Config(config);
    configInstance.validate();
    
    return configInstance;
  }

  public get app(): AppConfig['app'] {
    return this.config.app;
  }

  public get server(): ServerConfig {
    return this.config.server;
  }

  public get database(): DatabaseConfig | undefined {
    return this.config.database;
  }

  public get logging(): LoggingConfig {
    return this.config.logging;
  }

  public get apiKeys(): Record<string, string> {
    return this.config.apiKeys;
  }

  public isProduction(): boolean {
    return this.config.app.environment === 'production';
  }

  public isDevelopment(): boolean {
    return this.config.app.environment === 'development';
  }

  public isTest(): boolean {
    return this.config.app.environment === 'test';
  }

  private validate(): void {
    if (this.config.server.port <= 0 || this.config.server.port > 65535) {
      throw new Error('Invalid server port');
    }

    if (!this.config.app.name.trim()) {
      throw new Error('App name is required');
    }

    if (this.config.database) {
      if (!this.config.database.host || !this.config.database.database) {
        throw new Error('Database host and database name are required');
      }
    }
  }

  private static parseDatabaseUrl(url: string): DatabaseConfig {
    // Simplified database URL parsing
    // In real implementation, use a proper URL parser
    return {
      host: 'localhost',
      port: 5432,
      database: 'app_db',
      username: 'user',
      password: 'password',
    };
  }

  private static parseApiKeys(keysString?: string): Record<string, string> {
    if (!keysString) return {};
    
    const keys: Record<string, string> = {};
    keysString.split(',').forEach(pair => {
      const [key, value] = pair.split(':');
      if (key && value) {
        keys[key.trim()] = value.trim();
      }
    });
    
    return keys;
  }
}
'''

        application_ts = '''/**
 * Main application class
 * Orchestrates application lifecycle and dependency management
 */

import { Config } from '../config/Config';
import { Logger } from '../utils/Logger';
import { ErrorHandler } from '../errors/ErrorHandler';
import { Metrics } from '../utils/Metrics';

export class Application {
  private readonly logger = Logger.create('Application');
  private readonly metrics = new Metrics();
  private readonly errorHandler = new ErrorHandler();
  private isStarted = false;

  constructor(private readonly config: Config) {}

  public async start(): Promise<void> {
    if (this.isStarted) {
      throw new Error('Application is already started');
    }

    try {
      this.logger.info(`Starting ${this.config.app.name} v${this.config.app.version}`);
      this.logger.info(`Environment: ${this.config.app.environment}`);

      // Initialize core components
      await this.initializeErrorHandling();
      await this.initializeMetrics();
      
      // Initialize business components
      await this.initializeServices();
      
      // Start server if configured
      if (this.config.server) {
        await this.startServer();
      }

      this.isStarted = true;
      this.metrics.increment('application.starts');
      
      this.logger.info('Application started successfully');
    } catch (error) {
      this.logger.error('Failed to start application:', error);
      throw error;
    }
  }

  public async stop(): Promise<void> {
    if (!this.isStarted) {
      return;
    }

    try {
      this.logger.info('Stopping application...');

      // Stop services in reverse order
      await this.stopServer();
      await this.cleanupServices();
      await this.cleanupMetrics();

      this.isStarted = false;
      this.metrics.increment('application.stops');
      
      this.logger.info('Application stopped successfully');
    } catch (error) {
      this.logger.error('Error stopping application:', error);
      throw error;
    }
  }

  public getMetrics(): Record<string, number> {
    return this.metrics.getAll();
  }

  public isHealthy(): boolean {
    // TODO: Implement health checks
    return this.isStarted;
  }

  private async initializeErrorHandling(): Promise<void> {
    this.errorHandler.setup();
    this.logger.debug('Error handling initialized');
  }

  private async initializeMetrics(): Promise<void> {
    this.metrics.initialize();
    this.logger.debug('Metrics initialized');
  }

  private async initializeServices(): Promise<void> {
    // TODO: Initialize business services
    this.logger.debug('Services initialized');
  }

  private async startServer(): Promise<void> {
    // TODO: Start HTTP server if this is a web application
    this.logger.info(`Server listening on ${this.config.server.host}:${this.config.server.port}`);
  }

  private async stopServer(): Promise<void> {
    // TODO: Stop HTTP server gracefully
    this.logger.debug('Server stopped');
  }

  private async cleanupServices(): Promise<void> {
    // TODO: Cleanup business services
    this.logger.debug('Services cleaned up');
  }

  private async cleanupMetrics(): Promise<void> {
    this.metrics.cleanup();
    this.logger.debug('Metrics cleaned up');
  }
}
'''

        logger_ts = '''/**
 * Logging utility
 * Provides structured logging with different levels and formats
 */

import { LoggingConfig } from '../config/Config';

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  data?: any;
  error?: Error;
}

export class Logger {
  private static globalConfig: LoggingConfig = {
    level: 'info',
    format: 'pretty',
  };

  private constructor(private readonly component: string) {}

  public static create(component: string): Logger {
    return new Logger(component);
  }

  public static configure(config: LoggingConfig): void {
    Logger.globalConfig = config;
  }

  public debug(message: string, data?: any): void {
    this.log('debug', message, data);
  }

  public info(message: string, data?: any): void {
    this.log('info', message, data);
  }

  public warn(message: string, data?: any, error?: Error): void {
    this.log('warn', message, data, error);
  }

  public error(message: string, error?: Error | any, data?: any): void {
    if (error instanceof Error) {
      this.log('error', message, data, error);
    } else {
      // If second parameter is not an Error, treat it as data
      this.log('error', message, error, undefined);
    }
  }

  private log(level: LogLevel, message: string, data?: any, error?: Error): void {
    if (!this.shouldLog(level)) {
      return;
    }

    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      component: this.component,
      message,
      data,
      error,
    };

    if (Logger.globalConfig.format === 'json') {
      console.log(JSON.stringify(entry));
    } else {
      this.logPretty(entry);
    }
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: Record<LogLevel, number> = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3,
    };

    return levels[level] >= levels[Logger.globalConfig.level];
  }

  private logPretty(entry: LogEntry): void {
    const timestamp = entry.timestamp.substring(11, 19); // HH:MM:SS
    const levelStr = entry.level.toUpperCase().padEnd(5);
    const component = `[${entry.component}]`.padEnd(12);
    
    let output = `${timestamp} ${levelStr} ${component} ${entry.message}`;
    
    if (entry.data) {
      output += ` ${JSON.stringify(entry.data)}`;
    }
    
    if (entry.error) {
      output += `\\n  Error: ${entry.error.message}`;
      if (entry.error.stack) {
        output += `\\n  Stack: ${entry.error.stack}`;
      }
    }

    // Use appropriate console method based on level
    switch (entry.level) {
      case 'debug':
        console.debug(output);
        break;
      case 'info':
        console.info(output);
        break;
      case 'warn':
        console.warn(output);
        break;
      case 'error':
        console.error(output);
        break;
    }
  }
}
'''

        errors_ts = '''/**
 * Error handling utilities
 * Provides structured error types and global error handling
 */

export enum ErrorCode {
  UNKNOWN = 'UNKNOWN',
  VALIDATION = 'VALIDATION',
  NOT_FOUND = 'NOT_FOUND',
  UNAUTHORIZED = 'UNAUTHORIZED',
  FORBIDDEN = 'FORBIDDEN',
  CONFLICT = 'CONFLICT',
  INTERNAL = 'INTERNAL',
  NETWORK = 'NETWORK',
  DATABASE = 'DATABASE',
  CONFIGURATION = 'CONFIGURATION',
}

export class AppError extends Error {
  public readonly code: ErrorCode;
  public readonly statusCode: number;
  public readonly isOperational: boolean;
  public readonly details?: Record<string, any>;

  constructor(
    message: string,
    code: ErrorCode = ErrorCode.UNKNOWN,
    statusCode: number = 500,
    isOperational: boolean = true,
    details?: Record<string, any>
  ) {
    super(message);
    
    this.name = this.constructor.name;
    this.code = code;
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.details = details;

    Error.captureStackTrace(this, this.constructor);
  }

  public toJSON(): Record<string, any> {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      statusCode: this.statusCode,
      details: this.details,
    };
  }

  public static validation(message: string, field?: string, details?: Record<string, any>): AppError {
    return new AppError(
      message,
      ErrorCode.VALIDATION,
      400,
      true,
      { field, ...details }
    );
  }

  public static notFound(resource: string, id?: string): AppError {
    return new AppError(
      `${resource} not found`,
      ErrorCode.NOT_FOUND,
      404,
      true,
      { resource, id }
    );
  }

  public static unauthorized(message: string = 'Unauthorized'): AppError {
    return new AppError(message, ErrorCode.UNAUTHORIZED, 401);
  }

  public static forbidden(message: string = 'Forbidden'): AppError {
    return new AppError(message, ErrorCode.FORBIDDEN, 403);
  }

  public static conflict(message: string, resource?: string): AppError {
    return new AppError(
      message,
      ErrorCode.CONFLICT,
      409,
      true,
      { resource }
    );
  }

  public static internal(message: string, originalError?: Error): AppError {
    return new AppError(
      message,
      ErrorCode.INTERNAL,
      500,
      false,
      { originalError: originalError?.message }
    );
  }
}

export class ErrorHandler {
  public setup(): void {
    process.on('uncaughtException', this.handleUncaughtException.bind(this));
    process.on('unhandledRejection', this.handleUnhandledRejection.bind(this));
  }

  public handleError(error: Error | AppError): void {
    console.error('Error handled:', error);

    if (error instanceof AppError) {
      if (!error.isOperational) {
        console.error('Non-operational error detected, shutting down...');
        process.exit(1);
      }
    }
  }

  private handleUncaughtException(error: Error): void {
    console.error('Uncaught Exception:', error);
    process.exit(1);
  }

  private handleUnhandledRejection(reason: any, promise: Promise<any>): void {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
  }
}
'''

        utils_ts = '''/**
 * Utility functions and helpers
 * Common utilities for the modernized application
 */

export class Utils {
  /**
   * Generate a unique ID based on timestamp
   */
  public static generateId(): string {
    const timestamp = Date.now();
    const random = Math.floor(Math.random() * 10000);
    return `id_${timestamp}_${random}`;
  }

  /**
   * Validate email format using regex
   */
  public static validateEmail(email: string): boolean {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Sleep for specified milliseconds
   */
  public static async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Retry operation with exponential backoff
   */
  public static async retryWithBackoff<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    baseDelayMs: number = 1000
  ): Promise<T> {
    let lastError: Error;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        
        if (attempt === maxRetries) {
          throw lastError;
        }
        
        const delay = baseDelayMs * Math.pow(2, attempt - 1);
        await Utils.sleep(delay);
      }
    }
    
    throw lastError!;
  }

  /**
   * Deep clone an object
   */
  public static deepClone<T>(obj: T): T {
    if (obj === null || typeof obj !== 'object') {
      return obj;
    }
    
    if (obj instanceof Date) {
      return new Date(obj.getTime()) as unknown as T;
    }
    
    if (Array.isArray(obj)) {
      return obj.map(item => Utils.deepClone(item)) as unknown as T;
    }
    
    const cloned = {} as T;
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        cloned[key] = Utils.deepClone(obj[key]);
      }
    }
    
    return cloned;
  }

  /**
   * Format bytes as human readable string
   */
  public static formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const threshold = 1024;
    
    if (bytes === 0) return '0 B';
    
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= threshold && unitIndex < units.length - 1) {
      size /= threshold;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  }

  /**
   * Debounce function calls
   */
  public static debounce<T extends (...args: any[]) => any>(
    func: T,
    waitMs: number
  ): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;
    
    return (...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func.apply(this, args), waitMs);
    };
  }

  /**
   * Throttle function calls
   */
  public static throttle<T extends (...args: any[]) => any>(
    func: T,
    limitMs: number
  ): (...args: Parameters<T>) => void {
    let inThrottle: boolean;
    
    return (...args: Parameters<T>) => {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limitMs);
      }
    };
  }
}

/**
 * Metrics collection utility
 */
export class Metrics {
  private counters = new Map<string, number>();
  private gauges = new Map<string, number>();
  private histograms = new Map<string, number[]>();

  public initialize(): void {
    // Initialize metrics collection
    this.counters.clear();
    this.gauges.clear();
    this.histograms.clear();
  }

  public increment(name: string, value: number = 1): void {
    const current = this.counters.get(name) || 0;
    this.counters.set(name, current + value);
  }

  public decrement(name: string, value: number = 1): void {
    this.increment(name, -value);
  }

  public gauge(name: string, value: number): void {
    this.gauges.set(name, value);
  }

  public histogram(name: string, value: number): void {
    if (!this.histograms.has(name)) {
      this.histograms.set(name, []);
    }
    this.histograms.get(name)!.push(value);
  }

  public getAll(): Record<string, number> {
    const metrics: Record<string, number> = {};
    
    // Add counters
    this.counters.forEach((value, key) => {
      metrics[`counter.${key}`] = value;
    });
    
    // Add gauges
    this.gauges.forEach((value, key) => {
      metrics[`gauge.${key}`] = value;
    });
    
    // Add histogram averages
    this.histograms.forEach((values, key) => {
      if (values.length > 0) {
        const average = values.reduce((a, b) => a + b, 0) / values.length;
        metrics[`histogram.${key}.avg`] = average;
        metrics[`histogram.${key}.count`] = values.length;
      }
    });
    
    return metrics;
  }

  public cleanup(): void {
    this.counters.clear();
    this.gauges.clear();
    this.histograms.clear();
  }
}
'''

        # Test files
        config_test_ts = '''/**
 * Tests for Config class
 */

import { Config } from '../src/config/Config';

describe('Config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetModules();
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  describe('load', () => {
    it('should load default configuration', async () => {
      const config = await Config.load();
      
      expect(config.app.name).toBe('PomegranteMuse Generated App');
      expect(config.app.version).toBe('0.1.0');
      expect(config.app.environment).toBe('development');
      expect(config.server.port).toBe(3000);
    });

    it('should load configuration from environment variables', async () => {
      process.env.APP_NAME = 'Test App';
      process.env.PORT = '8080';
      process.env.NODE_ENV = 'production';
      
      const config = await Config.load();
      
      expect(config.app.name).toBe('Test App');
      expect(config.server.port).toBe(8080);
      expect(config.app.environment).toBe('production');
    });

    it('should validate configuration', async () => {
      process.env.PORT = '-1';
      
      await expect(Config.load()).rejects.toThrow('Invalid server port');
    });
  });

  describe('helper methods', () => {
    it('should identify production environment', async () => {
      process.env.NODE_ENV = 'production';
      const config = await Config.load();
      
      expect(config.isProduction()).toBe(true);
      expect(config.isDevelopment()).toBe(false);
    });

    it('should identify development environment', async () => {
      process.env.NODE_ENV = 'development';
      const config = await Config.load();
      
      expect(config.isDevelopment()).toBe(true);
      expect(config.isProduction()).toBe(false);
    });
  });
});
'''

        utils_test_ts = '''/**
 * Tests for Utils class
 */

import { Utils } from '../src/utils/Utils';

describe('Utils', () => {
  describe('generateId', () => {
    it('should generate unique IDs', () => {
      const id1 = Utils.generateId();
      const id2 = Utils.generateId();
      
      expect(id1).not.toBe(id2);
      expect(id1).toMatch(/^id_\\d+_\\d+$/);
    });
  });

  describe('validateEmail', () => {
    it('should validate correct emails', () => {
      expect(Utils.validateEmail('user@example.com')).toBe(true);
      expect(Utils.validateEmail('test.user@domain.co.uk')).toBe(true);
    });

    it('should reject invalid emails', () => {
      expect(Utils.validateEmail('invalid_email')).toBe(false);
      expect(Utils.validateEmail('@domain.com')).toBe(false);
      expect(Utils.validateEmail('user@')).toBe(false);
    });
  });

  describe('sleep', () => {
    it('should sleep for specified time', async () => {
      const start = Date.now();
      await Utils.sleep(100);
      const end = Date.now();
      
      expect(end - start).toBeGreaterThanOrEqual(90);
    });
  });

  describe('retryWithBackoff', () => {
    it('should retry failed operations', async () => {
      let attempts = 0;
      const operation = async () => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Temporary failure');
        }
        return 'success';
      };

      const result = await Utils.retryWithBackoff(operation, 5, 10);
      
      expect(result).toBe('success');
      expect(attempts).toBe(3);
    });

    it('should fail after max retries', async () => {
      const operation = async () => {
        throw new Error('Always fails');
      };

      await expect(
        Utils.retryWithBackoff(operation, 2, 10)
      ).rejects.toThrow('Always fails');
    });
  });

  describe('deepClone', () => {
    it('should deep clone objects', () => {
      const original = {
        name: 'test',
        nested: { value: 42 },
        array: [1, 2, { inner: 'value' }]
      };

      const cloned = Utils.deepClone(original);
      
      expect(cloned).toEqual(original);
      expect(cloned).not.toBe(original);
      expect(cloned.nested).not.toBe(original.nested);
      expect(cloned.array).not.toBe(original.array);
    });
  });

  describe('formatBytes', () => {
    it('should format bytes correctly', () => {
      expect(Utils.formatBytes(0)).toBe('0 B');
      expect(Utils.formatBytes(500)).toBe('500.0 B');
      expect(Utils.formatBytes(1024)).toBe('1.0 KB');
      expect(Utils.formatBytes(1536)).toBe('1.5 KB');
      expect(Utils.formatBytes(1048576)).toBe('1.0 MB');
    });
  });
});
'''

        return {
            "package.json": json.dumps(package_json, indent=2),
            "tsconfig.json": json.dumps(tsconfig_json, indent=2),
            "jest.config.js": f"module.exports = {json.dumps(jest_config, indent=2)};",
            ".eslintrc.json": json.dumps(eslint_config, indent=2),
            ".prettierrc": json.dumps(prettier_config, indent=2),
            "src/index.ts": index_ts,
            "src/config/Config.ts": config_ts,
            "src/core/Application.ts": application_ts,
            "src/utils/Logger.ts": logger_ts,
            "src/errors/ErrorHandler.ts": errors_ts,
            "src/utils/Utils.ts": utils_ts,
            "tests/config/Config.test.ts": config_test_ts,
            "tests/utils/Utils.test.ts": utils_test_ts,
            ".gitignore": self._generate_gitignore(),
            "README.md": self._generate_readme(context),
            "Dockerfile": self._generate_dockerfile(project_name),
            ".dockerignore": self._generate_dockerignore(),
        }
    
    def generate_function(self, function_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a TypeScript function"""
        name = function_info.get("name", "unnamedFunction")
        params = function_info.get("parameters", [])
        return_type = function_info.get("return_type", "void")
        is_async = function_info.get("is_async", False)
        visibility = function_info.get("visibility", "public")
        description = function_info.get("description", "")
        
        # Convert parameters
        ts_params = []
        for param in params:
            param_name = param.get("name", "param")
            param_type = param.get("type", "any")
            optional = param.get("optional", False)
            ts_type = self._convert_type(param_type)
            
            param_str = f"{param_name}{'?' if optional else ''}: {ts_type}"
            ts_params.append(param_str)
        
        # Convert return type
        ts_return_type = self._convert_type(return_type)
        if is_async and not ts_return_type.startswith("Promise"):
            ts_return_type = f"Promise<{ts_return_type}>"
        
        # Visibility
        vis = "public " if visibility == "public" else "private " if visibility == "private" else ""
        
        # Async keyword
        async_kw = "async " if is_async else ""
        
        # Generate documentation
        doc = ""
        if description:
            doc = f"  /**\n   * {description}\n   */\n"
        
        # Generate function signature
        params_str = ", ".join(ts_params)
        
        # Generate function body
        body = self._generate_function_body(function_info, ts_return_type, is_async)
        
        return f'''{doc}{vis}{async_kw}{name}({params_str}): {ts_return_type} {{
{body}
  }}'''
    
    def generate_class(self, class_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a TypeScript class"""
        name = class_info.get("name", "UnnamedClass")
        fields = class_info.get("fields", [])
        methods = class_info.get("methods", [])
        interfaces = class_info.get("implements", [])
        extends = class_info.get("extends")
        description = class_info.get("description", "")
        is_abstract = class_info.get("abstract", False)
        
        # Generate class properties
        properties = []
        constructor_params = []
        constructor_assignments = []
        
        for field in fields:
            field_name = field.get("name", "field")
            field_type = field.get("type", "any")
            field_vis = field.get("visibility", "private")
            readonly = field.get("readonly", False)
            optional = field.get("optional", False)
            
            ts_type = self._convert_type(field_type)
            vis_keyword = "public " if field_vis == "public" else "private " if field_vis == "private" else "protected "
            readonly_keyword = "readonly " if readonly else ""
            optional_marker = "?" if optional else ""
            
            properties.append(f"  {vis_keyword}{readonly_keyword}{field_name}{optional_marker}: {ts_type};")
            
            # Add to constructor if not readonly or has default
            if not readonly:
                constructor_params.append(f"{field_name}: {ts_type}")
                constructor_assignments.append(f"    this.{field_name} = {field_name};")
        
        # Generate constructor
        constructor = ""
        if constructor_params:
            constructor = f'''
  constructor({", ".join(constructor_params)}) {{
{chr(10).join(constructor_assignments)}
  }}'''
        
        # Generate methods
        class_methods = []
        for method in methods:
            ts_method = self.generate_function(method, context)
            # Add proper indentation
            indented_method = "\n".join(f"  {line}" for line in ts_method.split("\n"))
            class_methods.append(indented_method)
        
        # Class declaration parts
        abstract_keyword = "abstract " if is_abstract else ""
        extends_clause = f" extends {extends}" if extends else ""
        implements_clause = f" implements {', '.join(interfaces)}" if interfaces else ""
        
        # Documentation
        doc = ""
        if description:
            doc = f"/**\n * {description}\n */\n"
        
        # Build class
        properties_str = "\n".join(properties) if properties else ""
        methods_str = "\n\n".join(class_methods) if class_methods else ""
        
        class_body = properties_str
        if constructor:
            class_body += constructor
        if methods_str:
            class_body += "\n\n" + methods_str
        
        return f'''{doc}{abstract_keyword}class {name}{extends_clause}{implements_clause} {{
{class_body}
}}'''
    
    def generate_module(self, module_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate a TypeScript module"""
        description = module_info.get("description", "")
        functions = module_info.get("functions", [])
        classes = module_info.get("classes", [])
        interfaces = module_info.get("interfaces", [])
        types = module_info.get("types", [])
        imports = module_info.get("imports", [])
        exports = module_info.get("exports", [])
        
        # Module header
        header = f'''/**
 * {description}
 */

'''
        
        # Imports
        imports_section = ""
        if imports:
            ts_imports = []
            for imp in imports:
                ts_imports.append(self._convert_import(imp))
            imports_section = "\n".join(ts_imports) + "\n\n"
        
        # Type definitions
        types_section = ""
        for type_def in types:
            types_section += self._generate_type_definition(type_def) + "\n\n"
        
        # Interfaces
        interfaces_section = ""
        for interface in interfaces:
            interfaces_section += self._generate_interface(interface) + "\n\n"
        
        # Classes
        classes_section = ""
        for class_info in classes:
            classes_section += self.generate_class(class_info, context) + "\n\n"
        
        # Functions
        functions_section = ""
        for func_info in functions:
            functions_section += self.generate_function(func_info, context) + "\n\n"
        
        # Exports
        exports_section = ""
        if exports:
            exports_section = f"export {{ {', '.join(exports)} }};\n"
        
        return header + imports_section + types_section + interfaces_section + classes_section + functions_section + exports_section
    
    def generate_error_handling(self, error_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate TypeScript error handling code"""
        error_type = error_info.get("type", "generic")
        message = error_info.get("message", "An error occurred")
        
        if error_type == "validation":
            field = error_info.get("field", "unknown")
            return f'AppError.validation("{message}", "{field}")'
        elif error_type == "not_found":
            resource = error_info.get("resource", "resource")
            return f'AppError.notFound("{resource}")'
        elif error_type == "unauthorized":
            return f'AppError.unauthorized("{message}")'
        else:
            return f'new Error("{message}")'
    
    def generate_async_code(self, async_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate TypeScript async code"""
        operation = async_info.get("operation", "")
        
        if "http_request" in operation.lower():
            return '''const response = await fetch(url, {
  method: 'GET',
  headers: {
    'Content-Type': 'application/json',
  },
});

if (!response.ok) {
  throw new Error(`HTTP error! status: ${response.status}`);
}

const data = await response.json();
return data;'''
        elif "file_read" in operation.lower():
            return '''import { promises as fs } from 'fs';

const content = await fs.readFile(filePath, 'utf-8');
return content;'''
        elif "database" in operation.lower():
            return '''const result = await this.database.query(
  'SELECT * FROM table WHERE condition = $1',
  [parameter]
);
return result.rows;'''
        else:
            return '''// TODO: Implement async operation
await Utils.sleep(100);
return result;'''
    
    def generate_tests(self, test_info: Dict[str, Any], context: CodeGenerationContext) -> str:
        """Generate TypeScript test code"""
        test_name = test_info.get("name", "test function")
        is_async = test_info.get("is_async", False)
        
        # Clean test name
        clean_name = test_name.replace("test_", "").replace("_", " ")
        
        test_func = "it" if not is_async else "it"
        
        return f'''  {test_func}('should {clean_name}', async () => {{
    // TODO: Implement test
    expect(true).toBe(true);
  }});'''
    
    def get_migration_strategies(self, source_analysis: Dict[str, Any]) -> List[MigrationStrategy]:
        """Get TypeScript-specific migration strategies"""
        strategies = []
        
        # Gradual TypeScript adoption
        strategies.append(MigrationStrategy(
            name="Gradual TypeScript Migration",
            description="Convert JavaScript to TypeScript incrementally with strict type checking",
            approach="incremental",
            complexity="low",
            timeline_estimate="2-4 months",
            risks=["Type definition complexity", "Build tooling setup"],
            benefits=["Type safety", "Better IDE support", "Refactoring safety"],
            prerequisites=["Modern build setup", "Team TypeScript training"],
            requires_manual_review=True,
            supports_gradual_migration=True,
            maintains_performance=True,
            preserves_architecture=True
        ))
        
        # Modern web application
        strategies.append(MigrationStrategy(
            name="Modern Web Stack",
            description="Full rewrite with modern TypeScript, async/await, and type-safe APIs",
            approach="rewrite",
            complexity="medium",
            timeline_estimate="3-6 months",
            risks=["Framework learning curve", "API design complexity"],
            benefits=["Type safety", "Modern async patterns", "Excellent tooling"],
            prerequisites=["Team TypeScript experience", "API design review"],
            requires_manual_review=True,
            supports_gradual_migration=False,
            maintains_performance=True,
            preserves_architecture=False
        ))
        
        return strategies
    
    def get_best_practices(self, context: CodeGenerationContext) -> List[str]:
        """Get TypeScript best practices"""
        return [
            "Use strict TypeScript configuration with noImplicitAny",
            "Prefer interfaces over type aliases for object shapes",
            "Use union types instead of any when possible",
            "Implement proper error handling with typed errors",
            "Use async/await instead of Promise.then() chains",
            "Organize code with barrel exports (index.ts files)",
            "Use const assertions for immutable data",
            "Prefer composition over inheritance",
            "Write comprehensive unit tests with Jest",
            "Use ESLint and Prettier for consistent code style"
        ]
    
    def _convert_type(self, original_type: str) -> str:
        """Convert generic type to TypeScript type"""
        if original_type in self.type_mappings:
            return self.type_mappings[original_type]
        
        # Handle generic types
        if original_type.startswith("List<") or original_type.startswith("Array<"):
            inner_type = original_type[5:-1] if original_type.startswith("List<") else original_type[6:-1]
            return f"Array<{self._convert_type(inner_type)}>"
        elif original_type.startswith("Map<"):
            inner = original_type[4:-1]
            if "," in inner:
                key_type, value_type = inner.split(",", 1)
                return f"Map<{self._convert_type(key_type.strip())}, {self._convert_type(value_type.strip())}>"
        elif original_type.startswith("Optional<"):
            inner_type = original_type[9:-1]
            return f"{self._convert_type(inner_type)} | undefined"
        
        # Default to any for unknown types (should be avoided)
        return "any"
    
    def _convert_import(self, import_statement: str) -> str:
        """Convert import statement to TypeScript import"""
        # This is a simplified conversion
        if isinstance(import_statement, dict):
            module = import_statement.get("module", "")
            imports = import_statement.get("imports", [])
            
            if imports:
                imports_str = f"{{ {', '.join(imports)} }}"
            else:
                imports_str = "*"
            
            return f"import {imports_str} from '{module}';"
        else:
            return f"import {{ }} from '{import_statement}';"
    
    def _sanitize_package_name(self, name: str) -> str:
        """Sanitize package name for npm"""
        return name.lower().replace(" ", "-").replace("_", "-")
    
    def _get_dependencies(self, context: CodeGenerationContext) -> Dict[str, str]:
        """Get runtime dependencies based on context"""
        base_deps = {}
        
        if context.domain == "web":
            base_deps.update({
                "express": "^4.18.2",
                "helmet": "^7.1.0",
                "cors": "^2.8.5",
                "morgan": "^1.10.0",
                "compression": "^1.7.4"
            })
        
        if context.domain == "cli":
            base_deps.update({
                "commander": "^11.1.0",
                "inquirer": "^9.2.0"
            })
        
        return base_deps
    
    def _get_dev_dependencies(self, context: CodeGenerationContext) -> Dict[str, str]:
        """Get development dependencies"""
        return {
            "@types/node": "^20.10.0",
            "@typescript-eslint/eslint-plugin": "^6.13.0",
            "@typescript-eslint/parser": "^6.13.0",
            "eslint": "^8.54.0",
            "eslint-config-prettier": "^9.0.0",
            "jest": "^29.7.0",
            "@types/jest": "^29.5.8",
            "ts-jest": "^29.1.1",
            "typescript": "^5.3.2",
            "ts-node": "^10.9.1",
            "ts-node-dev": "^2.0.0",
            "prettier": "^3.1.0",
            "rimraf": "^5.0.5",
            "copyfiles": "^2.4.1"
        }
    
    def _generate_function_body(self, function_info: Dict[str, Any], return_type: str, is_async: bool) -> str:
        """Generate function body"""
        if is_async:
            if "void" in return_type:
                return "    // TODO: Implement async function"
            else:
                return "    // TODO: Implement async function\n    return null as any;"
        else:
            if "void" in return_type:
                return "    // TODO: Implement function"
            else:
                return "    // TODO: Implement function\n    return null as any;"
    
    def _generate_interface(self, interface_info: Dict[str, Any]) -> str:
        """Generate TypeScript interface"""
        name = interface_info.get("name", "UnnamedInterface")
        properties = interface_info.get("properties", [])
        extends = interface_info.get("extends", [])
        description = interface_info.get("description", "")
        
        # Generate properties
        props = []
        for prop in properties:
            prop_name = prop.get("name", "prop")
            prop_type = prop.get("type", "any")
            optional = prop.get("optional", False)
            readonly = prop.get("readonly", False)
            
            ts_type = self._convert_type(prop_type)
            readonly_keyword = "readonly " if readonly else ""
            optional_marker = "?" if optional else ""
            
            props.append(f"  {readonly_keyword}{prop_name}{optional_marker}: {ts_type};")
        
        # Extends clause
        extends_clause = f" extends {', '.join(extends)}" if extends else ""
        
        # Documentation
        doc = ""
        if description:
            doc = f"/**\n * {description}\n */\n"
        
        properties_str = "\n".join(props) if props else "  // Empty interface"
        
        return f'''{doc}interface {name}{extends_clause} {{
{properties_str}
}}'''
    
    def _generate_type_definition(self, type_info: Dict[str, Any]) -> str:
        """Generate TypeScript type definition"""
        name = type_info.get("name", "UnnamedType")
        definition = type_info.get("definition", "any")
        description = type_info.get("description", "")
        
        # Documentation
        doc = ""
        if description:
            doc = f"/**\n * {description}\n */\n"
        
        return f"{doc}type {name} = {definition};"
    
    def _generate_gitignore(self) -> str:
        """Generate .gitignore for TypeScript project"""
        return '''# Dependencies
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Build output
dist/
build/
*.tsbuildinfo

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# Logs
logs
*.log

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache
'''
    
    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore for TypeScript project"""
        return '''node_modules
npm-debug.log
coverage
.git
.gitignore
README.md
.env
.nyc_output
'''
    
    def _generate_dockerfile(self, project_name: str) -> str:
        """Generate Dockerfile for TypeScript project"""
        return f'''# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies)
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001

# Change ownership of the working directory
RUN chown -R nodejs:nodejs /app
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
  CMD node dist/health-check.js || exit 1

# Start the application
CMD ["node", "dist/index.js"]
'''
    
    def _generate_readme(self, context: CodeGenerationContext) -> str:
        """Generate README.md for TypeScript project"""
        project_name = self._sanitize_package_name(context.domain)
        return f'''# {context.domain.title()} - TypeScript Migration

This project was generated by PomegranteMuse, migrating from {context.source_language} to TypeScript.

## Features

- Type-safe TypeScript with strict configuration
- Modern async/await patterns
- Comprehensive error handling
- Structured logging
- Configuration management
- Unit testing with Jest
- Code formatting with Prettier
- Linting with ESLint

## Getting Started

### Prerequisites

- Node.js 18 or later
- npm 9 or later

### Installation

```bash
npm install
```

### Development

```bash
# Start in development mode with hot reload
npm run dev

# Build the project
npm run build

# Run in production mode
npm start
```

### Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

### Code Quality

```bash
# Lint code
npm run lint

# Fix linting issues
npm run lint:fix

# Format code
npm run format

# Type check without emitting
npm run typecheck
```

## Project Structure

```
src/
├── config/           # Configuration management
├── core/            # Core application logic
├── errors/          # Error handling
├── utils/           # Utility functions
└── index.ts         # Application entry point

tests/               # Test files
dist/                # Compiled output
```

## Docker

Build and run with Docker:

```bash
# Build image
docker build -t {project_name} .

# Run container
docker run -p 3000:3000 {project_name}
```

## Environment Variables

- `NODE_ENV` - Environment (development/test/production)
- `PORT` - Server port (default: 3000)
- `LOG_LEVEL` - Logging level (debug/info/warn/error)
- `DATABASE_URL` - Database connection string

## Generated by

PomegranteMuse - Universal Code Modernization Platform  
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
'''