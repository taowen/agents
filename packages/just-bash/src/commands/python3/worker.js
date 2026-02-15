var __require = /* @__PURE__ */ ((x) =>
  typeof require !== "undefined"
    ? require
    : typeof Proxy !== "undefined"
      ? new Proxy(x, {
          get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
        })
      : x)(function (x) {
  if (typeof require !== "undefined") return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x + '" is not supported');
});

// src/commands/python3/worker.ts
import { createRequire } from "node:module";
import { dirname } from "node:path";
import { parentPort, workerData } from "node:worker_threads";
import { loadPyodide } from "pyodide";

// src/security/blocked-globals.ts
function getBlockedGlobals() {
  const globals = [
    // Direct code execution vectors
    {
      prop: "Function",
      target: globalThis,
      violationType: "function_constructor",
      strategy: "throw",
      reason: "Function constructor allows arbitrary code execution"
    },
    {
      prop: "eval",
      target: globalThis,
      violationType: "eval",
      strategy: "throw",
      reason: "eval() allows arbitrary code execution"
    },
    // Timer functions with string argument allow code execution
    {
      prop: "setTimeout",
      target: globalThis,
      violationType: "setTimeout",
      strategy: "throw",
      reason: "setTimeout with string argument allows code execution"
    },
    {
      prop: "setInterval",
      target: globalThis,
      violationType: "setInterval",
      strategy: "throw",
      reason: "setInterval with string argument allows code execution"
    },
    {
      prop: "setImmediate",
      target: globalThis,
      violationType: "setImmediate",
      strategy: "throw",
      reason: "setImmediate could be used to escape sandbox context"
    },
    // Note: We intentionally do NOT block `process` entirely because:
    // 1. Node.js internals (Promise resolution, etc.) use process.nextTick
    // 2. Blocking process entirely breaks normal async operation
    // 3. The primary code execution vectors (Function, eval) are already blocked
    // However, we DO block specific dangerous process properties.
    {
      prop: "env",
      target: process,
      violationType: "process_env",
      strategy: "throw",
      reason: "process.env could leak sensitive environment variables"
    },
    {
      prop: "binding",
      target: process,
      violationType: "process_binding",
      strategy: "throw",
      reason: "process.binding provides access to native Node.js modules"
    },
    {
      prop: "_linkedBinding",
      target: process,
      violationType: "process_binding",
      strategy: "throw",
      reason: "process._linkedBinding provides access to native Node.js modules"
    },
    {
      prop: "dlopen",
      target: process,
      violationType: "process_dlopen",
      strategy: "throw",
      reason: "process.dlopen allows loading native addons"
    },
    // Note: process.mainModule is handled specially in defense-in-depth-box.ts
    // and worker-defense-in-depth.ts because it may be undefined in ESM contexts
    // but we still want to block both reading and setting it.
    // We also don't block `require` because:
    // 1. It may not exist in all environments (ESM)
    // 2. import() is the modern escape vector and can't be blocked this way
    // Reference leak vectors
    {
      prop: "WeakRef",
      target: globalThis,
      violationType: "weak_ref",
      strategy: "throw",
      reason: "WeakRef could be used to leak references outside sandbox"
    },
    {
      prop: "FinalizationRegistry",
      target: globalThis,
      violationType: "finalization_registry",
      strategy: "throw",
      reason:
        "FinalizationRegistry could be used to leak references outside sandbox"
    },
    // Introspection/interception vectors (freeze instead of throw)
    {
      prop: "Reflect",
      target: globalThis,
      violationType: "reflect",
      strategy: "freeze",
      reason: "Reflect provides introspection capabilities"
    },
    {
      prop: "Proxy",
      target: globalThis,
      violationType: "proxy",
      strategy: "throw",
      reason: "Proxy allows intercepting and modifying object behavior"
    },
    // WebAssembly allows arbitrary code execution
    {
      prop: "WebAssembly",
      target: globalThis,
      violationType: "webassembly",
      strategy: "throw",
      reason: "WebAssembly allows executing arbitrary compiled code"
    },
    // SharedArrayBuffer and Atomics can enable side-channel attacks
    {
      prop: "SharedArrayBuffer",
      target: globalThis,
      violationType: "shared_array_buffer",
      strategy: "throw",
      reason:
        "SharedArrayBuffer could enable side-channel communication or timing attacks"
    },
    {
      prop: "Atomics",
      target: globalThis,
      violationType: "atomics",
      strategy: "throw",
      reason:
        "Atomics could enable side-channel communication or timing attacks"
    }
    // Note: Error.prepareStackTrace is handled specially in defense-in-depth-box.ts
    // because we only want to block SETTING it, not reading (V8 reads it internally)
  ];
  try {
    const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
    if (AsyncFunction && AsyncFunction !== Function) {
      globals.push({
        prop: "constructor",
        target: Object.getPrototypeOf(async () => {}),
        violationType: "async_function_constructor",
        strategy: "throw",
        reason:
          "AsyncFunction constructor allows arbitrary async code execution"
      });
    }
  } catch {}
  try {
    const GeneratorFunction = Object.getPrototypeOf(
      function* () {}
    ).constructor;
    if (GeneratorFunction && GeneratorFunction !== Function) {
      globals.push({
        prop: "constructor",
        target: Object.getPrototypeOf(function* () {}),
        violationType: "generator_function_constructor",
        strategy: "throw",
        reason:
          "GeneratorFunction constructor allows arbitrary generator code execution"
      });
    }
  } catch {}
  try {
    const AsyncGeneratorFunction = Object.getPrototypeOf(
      async function* () {}
    ).constructor;
    if (
      AsyncGeneratorFunction &&
      AsyncGeneratorFunction !== Function &&
      AsyncGeneratorFunction !==
        Object.getPrototypeOf(async () => {}).constructor
    ) {
      globals.push({
        prop: "constructor",
        target: Object.getPrototypeOf(async function* () {}),
        violationType: "async_generator_function_constructor",
        strategy: "throw",
        reason:
          "AsyncGeneratorFunction constructor allows arbitrary async generator code execution"
      });
    }
  } catch {}
  return globals.filter((g) => {
    try {
      return g.target[g.prop] !== void 0;
    } catch {
      return false;
    }
  });
}

// src/security/defense-in-depth-box.ts
var IS_BROWSER = typeof __BROWSER__ !== "undefined" && __BROWSER__;
var AsyncLocalStorageClass = null;
if (!IS_BROWSER) {
  try {
    const { createRequire: createRequire2 } = await import("node:module");
    const require3 = createRequire2(import.meta.url);
    const asyncHooks = require3("node:async_hooks");
    AsyncLocalStorageClass = asyncHooks.AsyncLocalStorage;
  } catch (e) {
    console.debug(
      "[DefenseInDepthBox] AsyncLocalStorage not available, defense-in-depth disabled:",
      e instanceof Error ? e.message : e
    );
  }
}
var executionContext =
  !IS_BROWSER && AsyncLocalStorageClass ? new AsyncLocalStorageClass() : null;

// src/security/worker-defense-in-depth.ts
var DEFENSE_IN_DEPTH_NOTICE =
  "\n\nThis is a defense-in-depth measure and indicates a bug in just-bash. Please report this at security@vercel.com";
var WorkerSecurityViolationError = class extends Error {
  constructor(message, violation) {
    super(message + DEFENSE_IN_DEPTH_NOTICE);
    this.violation = violation;
    this.name = "WorkerSecurityViolationError";
  }
};
var MAX_STORED_VIOLATIONS = 1e3;
function generateExecutionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 3) | 8;
    return v.toString(16);
  });
}
var WorkerDefenseInDepth = class {
  config;
  isActivated = false;
  originalDescriptors = [];
  violations = [];
  executionId;
  /**
   * Original Proxy constructor, captured before patching.
   * This is captured at instance creation time to ensure we get the unpatched version.
   */
  originalProxy;
  /**
   * Recursion guard to prevent infinite loops when proxy traps trigger
   * code that accesses the same proxied object (e.g., process.env).
   */
  inTrap = false;
  /**
   * Create and activate the worker defense layer.
   *
   * @param config - Configuration for the defense layer
   */
  constructor(config) {
    this.originalProxy = Proxy;
    this.config = config;
    this.executionId = generateExecutionId();
    if (config.enabled !== false) {
      this.activate();
    }
  }
  /**
   * Get statistics about the defense layer.
   */
  getStats() {
    return {
      violationsBlocked: this.violations.length,
      violations: [...this.violations],
      isActive: this.isActivated
    };
  }
  /**
   * Clear stored violations. Useful for testing.
   */
  clearViolations() {
    this.violations = [];
  }
  /**
   * Get the execution ID for this worker.
   */
  getExecutionId() {
    return this.executionId;
  }
  /**
   * Deactivate the defense layer and restore original globals.
   * Typically only needed for testing.
   */
  deactivate() {
    if (!this.isActivated) {
      return;
    }
    this.restorePatches();
    this.isActivated = false;
  }
  /**
   * Activate the defense layer by applying patches.
   */
  activate() {
    if (this.isActivated) {
      return;
    }
    this.applyPatches();
    this.isActivated = true;
  }
  /**
   * Get a human-readable path for a target object and property.
   */
  getPathForTarget(target, prop) {
    if (target === globalThis) {
      return `globalThis.${prop}`;
    }
    if (typeof process !== "undefined" && target === process) {
      return `process.${prop}`;
    }
    if (target === Error) {
      return `Error.${prop}`;
    }
    if (target === Function.prototype) {
      return `Function.prototype.${prop}`;
    }
    if (target === Object.prototype) {
      return `Object.prototype.${prop}`;
    }
    return `<object>.${prop}`;
  }
  /**
   * Record a violation and invoke the callback.
   * In worker context, blocking always happens (no audit mode context check).
   */
  recordViolation(type, path, message) {
    const violation = {
      timestamp: Date.now(),
      type,
      message,
      path,
      stack: new Error().stack,
      executionId: this.executionId
    };
    if (this.violations.length < MAX_STORED_VIOLATIONS) {
      this.violations.push(violation);
    }
    if (this.config.onViolation) {
      try {
        this.config.onViolation(violation);
      } catch (e) {
        console.debug(
          "[WorkerDefenseInDepth] onViolation callback threw:",
          e instanceof Error ? e.message : e
        );
      }
    }
    return violation;
  }
  /**
   * Create a blocking proxy for a function.
   * In worker context, always blocks (no context check needed).
   */
  // @banned-pattern-ignore: intentional use of Function type for security proxy
  createBlockingProxy(original, path, violationType) {
    const self = this;
    const auditMode = this.config.auditMode;
    return new this.originalProxy(original, {
      apply(target, thisArg, args) {
        const message = `${path} is blocked in worker context`;
        const violation = self.recordViolation(violationType, path, message);
        if (!auditMode) {
          throw new WorkerSecurityViolationError(message, violation);
        }
        return Reflect.apply(target, thisArg, args);
      },
      construct(target, args, newTarget) {
        const message = `${path} constructor is blocked in worker context`;
        const violation = self.recordViolation(violationType, path, message);
        if (!auditMode) {
          throw new WorkerSecurityViolationError(message, violation);
        }
        return Reflect.construct(target, args, newTarget);
      }
    });
  }
  /**
   * Create a blocking proxy for an object (blocks all property access).
   */
  createBlockingObjectProxy(original, path, violationType) {
    const self = this;
    const auditMode = this.config.auditMode;
    return new this.originalProxy(original, {
      get(target, prop, receiver) {
        if (self.inTrap) {
          return Reflect.get(target, prop, receiver);
        }
        self.inTrap = true;
        try {
          const fullPath = `${path}.${String(prop)}`;
          const message = `${fullPath} is blocked in worker context`;
          const violation = self.recordViolation(
            violationType,
            fullPath,
            message
          );
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.get(target, prop, receiver);
        } finally {
          self.inTrap = false;
        }
      },
      set(target, prop, value, receiver) {
        if (self.inTrap) {
          return Reflect.set(target, prop, value, receiver);
        }
        self.inTrap = true;
        try {
          const fullPath = `${path}.${String(prop)}`;
          const message = `${fullPath} modification is blocked in worker context`;
          const violation = self.recordViolation(
            violationType,
            fullPath,
            message
          );
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.set(target, prop, value, receiver);
        } finally {
          self.inTrap = false;
        }
      },
      ownKeys(target) {
        if (self.inTrap) {
          return Reflect.ownKeys(target);
        }
        self.inTrap = true;
        try {
          const message = `${path} enumeration is blocked in worker context`;
          const violation = self.recordViolation(violationType, path, message);
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.ownKeys(target);
        } finally {
          self.inTrap = false;
        }
      },
      getOwnPropertyDescriptor(target, prop) {
        if (self.inTrap) {
          return Reflect.getOwnPropertyDescriptor(target, prop);
        }
        self.inTrap = true;
        try {
          const fullPath = `${path}.${String(prop)}`;
          const message = `${fullPath} descriptor access is blocked in worker context`;
          const violation = self.recordViolation(
            violationType,
            fullPath,
            message
          );
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.getOwnPropertyDescriptor(target, prop);
        } finally {
          self.inTrap = false;
        }
      },
      has(target, prop) {
        if (self.inTrap) {
          return Reflect.has(target, prop);
        }
        self.inTrap = true;
        try {
          const fullPath = `${path}.${String(prop)}`;
          const message = `${fullPath} existence check is blocked in worker context`;
          const violation = self.recordViolation(
            violationType,
            fullPath,
            message
          );
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.has(target, prop);
        } finally {
          self.inTrap = false;
        }
      }
    });
  }
  /**
   * Apply security patches to dangerous globals.
   */
  applyPatches() {
    const blockedGlobals = getBlockedGlobals();
    const excludeTypes = new Set(this.config.excludeViolationTypes ?? []);
    for (const blocked of blockedGlobals) {
      if (excludeTypes.has(blocked.violationType)) {
        continue;
      }
      this.applyPatch(blocked);
    }
    if (!excludeTypes.has("function_constructor")) {
      this.protectConstructorChain(excludeTypes);
    }
    if (!excludeTypes.has("error_prepare_stack_trace")) {
      this.protectErrorPrepareStackTrace();
    }
    if (!excludeTypes.has("module_load")) {
      this.protectModuleLoad();
    }
    if (!excludeTypes.has("process_main_module")) {
      this.protectProcessMainModule();
    }
  }
  /**
   * Protect against .constructor.constructor escape vector.
   * @param excludeTypes - Set of violation types to skip
   */
  protectConstructorChain(excludeTypes) {
    let AsyncFunction = null;
    let GeneratorFunction = null;
    let AsyncGeneratorFunction = null;
    try {
      AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
    } catch {}
    try {
      GeneratorFunction = Object.getPrototypeOf(function* () {}).constructor;
    } catch {}
    try {
      AsyncGeneratorFunction = Object.getPrototypeOf(
        async function* () {}
      ).constructor;
    } catch {}
    this.patchPrototypeConstructor(
      Function.prototype,
      "Function.prototype.constructor",
      "function_constructor"
    );
    if (
      !excludeTypes.has("async_function_constructor") &&
      AsyncFunction &&
      AsyncFunction !== Function
    ) {
      this.patchPrototypeConstructor(
        AsyncFunction.prototype,
        "AsyncFunction.prototype.constructor",
        "async_function_constructor"
      );
    }
    if (
      !excludeTypes.has("generator_function_constructor") &&
      GeneratorFunction &&
      GeneratorFunction !== Function
    ) {
      this.patchPrototypeConstructor(
        GeneratorFunction.prototype,
        "GeneratorFunction.prototype.constructor",
        "generator_function_constructor"
      );
    }
    if (
      !excludeTypes.has("async_generator_function_constructor") &&
      AsyncGeneratorFunction &&
      AsyncGeneratorFunction !== Function &&
      AsyncGeneratorFunction !== AsyncFunction
    ) {
      this.patchPrototypeConstructor(
        AsyncGeneratorFunction.prototype,
        "AsyncGeneratorFunction.prototype.constructor",
        "async_generator_function_constructor"
      );
    }
  }
  /**
   * Protect Error.prepareStackTrace from being set.
   */
  protectErrorPrepareStackTrace() {
    const self = this;
    const auditMode = this.config.auditMode;
    try {
      const originalDescriptor = Object.getOwnPropertyDescriptor(
        Error,
        "prepareStackTrace"
      );
      this.originalDescriptors.push({
        target: Error,
        prop: "prepareStackTrace",
        descriptor: originalDescriptor
      });
      let currentValue = originalDescriptor?.value;
      Object.defineProperty(Error, "prepareStackTrace", {
        get() {
          return currentValue;
        },
        set(value) {
          const message =
            "Error.prepareStackTrace modification is blocked in worker context";
          const violation = self.recordViolation(
            "error_prepare_stack_trace",
            "Error.prepareStackTrace",
            message
          );
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          currentValue = value;
        },
        configurable: true
      });
    } catch {}
  }
  /**
   * Patch a prototype's constructor property.
   *
   * Returns a proxy that allows reading properties (like .name) but blocks
   * calling the constructor as a function (which would allow code execution).
   */
  patchPrototypeConstructor(prototype, path, violationType) {
    const self = this;
    const auditMode = this.config.auditMode;
    try {
      const originalDescriptor = Object.getOwnPropertyDescriptor(
        prototype,
        "constructor"
      );
      this.originalDescriptors.push({
        target: prototype,
        prop: "constructor",
        descriptor: originalDescriptor
      });
      const originalValue = originalDescriptor?.value;
      const constructorProxy =
        originalValue && typeof originalValue === "function"
          ? new this.originalProxy(originalValue, {
              apply(_target, _thisArg, _args) {
                const message = `${path} invocation is blocked in worker context`;
                const violation = self.recordViolation(
                  violationType,
                  path,
                  message
                );
                if (!auditMode) {
                  throw new WorkerSecurityViolationError(message, violation);
                }
                return void 0;
              },
              construct(_target, _args, _newTarget) {
                const message = `${path} construction is blocked in worker context`;
                const violation = self.recordViolation(
                  violationType,
                  path,
                  message
                );
                if (!auditMode) {
                  throw new WorkerSecurityViolationError(message, violation);
                }
                return {};
              },
              // Allow all property access (like .name, .prototype, etc.)
              get(target, prop, receiver) {
                return Reflect.get(target, prop, receiver);
              },
              getPrototypeOf(target) {
                return Reflect.getPrototypeOf(target);
              },
              has(target, prop) {
                return Reflect.has(target, prop);
              },
              ownKeys(target) {
                return Reflect.ownKeys(target);
              },
              getOwnPropertyDescriptor(target, prop) {
                return Reflect.getOwnPropertyDescriptor(target, prop);
              }
            })
          : originalValue;
      Object.defineProperty(prototype, "constructor", {
        get() {
          return constructorProxy;
        },
        set(value) {
          const message = `${path} modification is blocked in worker context`;
          const violation = self.recordViolation(violationType, path, message);
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          Object.defineProperty(this, "constructor", {
            value,
            writable: true,
            configurable: true
          });
        },
        configurable: true
      });
    } catch {}
  }
  /**
   * Protect process.mainModule from being accessed or set.
   *
   * The attack vector is:
   * ```
   * process.mainModule.require('child_process').execSync('whoami')
   * process.mainModule.constructor._load('vm')
   * ```
   *
   * process.mainModule may be undefined in ESM contexts but could exist in
   * CommonJS workers. We block both reading and setting.
   */
  protectProcessMainModule() {
    if (typeof process === "undefined") return;
    const self = this;
    const auditMode = this.config.auditMode;
    try {
      const originalDescriptor = Object.getOwnPropertyDescriptor(
        process,
        "mainModule"
      );
      this.originalDescriptors.push({
        target: process,
        prop: "mainModule",
        descriptor: originalDescriptor
      });
      const currentValue = originalDescriptor?.value;
      if (currentValue !== void 0) {
        Object.defineProperty(process, "mainModule", {
          get() {
            const message =
              "process.mainModule access is blocked in worker context";
            const violation = self.recordViolation(
              "process_main_module",
              "process.mainModule",
              message
            );
            if (!auditMode) {
              throw new WorkerSecurityViolationError(message, violation);
            }
            return currentValue;
          },
          set(value) {
            const message =
              "process.mainModule modification is blocked in worker context";
            const violation = self.recordViolation(
              "process_main_module",
              "process.mainModule",
              message
            );
            if (!auditMode) {
              throw new WorkerSecurityViolationError(message, violation);
            }
            Object.defineProperty(process, "mainModule", {
              value,
              writable: true,
              configurable: true
            });
          },
          configurable: true
        });
      }
    } catch {}
  }
  /**
   * Protect Module._load from being called.
   *
   * The attack vector is:
   * ```
   * module.constructor._load('child_process')
   * require.main.constructor._load('vm')
   * ```
   *
   * We access the Module class and replace _load with a blocking proxy.
   */
  protectModuleLoad() {
    const self = this;
    const auditMode = this.config.auditMode;
    try {
      let ModuleClass = null;
      if (typeof process !== "undefined") {
        const mainModule = process.mainModule;
        if (mainModule && typeof mainModule === "object") {
          ModuleClass = mainModule.constructor;
        }
      }
      if (
        !ModuleClass &&
        typeof __require !== "undefined" &&
        typeof __require.main !== "undefined"
      ) {
        ModuleClass = __require.main.constructor;
      }
      if (!ModuleClass || typeof ModuleClass._load !== "function") {
        return;
      }
      const original = ModuleClass._load;
      const descriptor = Object.getOwnPropertyDescriptor(ModuleClass, "_load");
      this.originalDescriptors.push({
        target: ModuleClass,
        prop: "_load",
        descriptor
      });
      const path = "Module._load";
      const proxy = new this.originalProxy(original, {
        apply(_target, _thisArg, _args) {
          const message = `${path} is blocked in worker context`;
          const violation = self.recordViolation("module_load", path, message);
          if (!auditMode) {
            throw new WorkerSecurityViolationError(message, violation);
          }
          return Reflect.apply(_target, _thisArg, _args);
        }
      });
      Object.defineProperty(ModuleClass, "_load", {
        value: proxy,
        writable: true,
        configurable: true
      });
    } catch {}
  }
  /**
   * Apply a single patch to a blocked global.
   */
  applyPatch(blocked) {
    const { target, prop, violationType, strategy } = blocked;
    try {
      const original = target[prop];
      if (original === void 0) {
        return;
      }
      const descriptor = Object.getOwnPropertyDescriptor(target, prop);
      this.originalDescriptors.push({ target, prop, descriptor });
      if (strategy === "freeze") {
        if (typeof original === "object" && original !== null) {
          Object.freeze(original);
        }
      } else {
        const path = this.getPathForTarget(target, prop);
        const proxy =
          typeof original === "function"
            ? this.createBlockingProxy(original, path, violationType)
            : this.createBlockingObjectProxy(original, path, violationType);
        Object.defineProperty(target, prop, {
          value: proxy,
          writable: true,
          configurable: true
        });
      }
    } catch {}
  }
  /**
   * Restore all original values.
   */
  restorePatches() {
    for (let i = this.originalDescriptors.length - 1; i >= 0; i--) {
      const { target, prop, descriptor } = this.originalDescriptors[i];
      try {
        if (descriptor) {
          Object.defineProperty(target, prop, descriptor);
        } else {
          delete target[prop];
        }
      } catch {}
    }
    this.originalDescriptors = [];
  }
};

// src/commands/python3/protocol.ts
var OpCode = {
  NOOP: 0,
  READ_FILE: 1,
  WRITE_FILE: 2,
  STAT: 3,
  READDIR: 4,
  MKDIR: 5,
  RM: 6,
  EXISTS: 7,
  APPEND_FILE: 8,
  SYMLINK: 9,
  READLINK: 10,
  LSTAT: 11,
  CHMOD: 12,
  REALPATH: 13,
  // Special operations for Python I/O
  WRITE_STDOUT: 100,
  WRITE_STDERR: 101,
  EXIT: 102,
  // HTTP operations
  HTTP_REQUEST: 200
};
var Status = {
  PENDING: 0,
  READY: 1,
  SUCCESS: 2,
  ERROR: 3
};
var ErrorCode = {
  NONE: 0,
  NOT_FOUND: 1,
  IS_DIRECTORY: 2,
  NOT_DIRECTORY: 3,
  EXISTS: 4,
  PERMISSION_DENIED: 5,
  INVALID_PATH: 6,
  IO_ERROR: 7,
  TIMEOUT: 8,
  NETWORK_ERROR: 9,
  NETWORK_NOT_CONFIGURED: 10
};
var Offset = {
  OP_CODE: 0,
  STATUS: 4,
  PATH_LENGTH: 8,
  DATA_LENGTH: 12,
  RESULT_LENGTH: 16,
  ERROR_CODE: 20,
  FLAGS: 24,
  MODE: 28,
  PATH_BUFFER: 32,
  DATA_BUFFER: 4128
  // 32 + 4096
};
var Size = {
  CONTROL_REGION: 32,
  PATH_BUFFER: 4096,
  DATA_BUFFER: 1048576,
  // 1MB (reduced from 16MB for faster tests)
  TOTAL: 1052704
  // 32 + 4096 + 1MB
};
var Flags = {
  NONE: 0,
  RECURSIVE: 1,
  FORCE: 2,
  MKDIR_RECURSIVE: 1
};
var StatLayout = {
  IS_FILE: 0,
  IS_DIRECTORY: 1,
  IS_SYMLINK: 2,
  MODE: 4,
  SIZE: 8,
  MTIME: 16,
  TOTAL: 24
};
var ProtocolBuffer = class {
  int32View;
  uint8View;
  dataView;
  constructor(buffer) {
    this.int32View = new Int32Array(buffer);
    this.uint8View = new Uint8Array(buffer);
    this.dataView = new DataView(buffer);
  }
  getOpCode() {
    return Atomics.load(this.int32View, Offset.OP_CODE / 4);
  }
  setOpCode(code) {
    Atomics.store(this.int32View, Offset.OP_CODE / 4, code);
  }
  getStatus() {
    return Atomics.load(this.int32View, Offset.STATUS / 4);
  }
  setStatus(status) {
    Atomics.store(this.int32View, Offset.STATUS / 4, status);
  }
  getPathLength() {
    return Atomics.load(this.int32View, Offset.PATH_LENGTH / 4);
  }
  setPathLength(length) {
    Atomics.store(this.int32View, Offset.PATH_LENGTH / 4, length);
  }
  getDataLength() {
    return Atomics.load(this.int32View, Offset.DATA_LENGTH / 4);
  }
  setDataLength(length) {
    Atomics.store(this.int32View, Offset.DATA_LENGTH / 4, length);
  }
  getResultLength() {
    return Atomics.load(this.int32View, Offset.RESULT_LENGTH / 4);
  }
  setResultLength(length) {
    Atomics.store(this.int32View, Offset.RESULT_LENGTH / 4, length);
  }
  getErrorCode() {
    return Atomics.load(this.int32View, Offset.ERROR_CODE / 4);
  }
  setErrorCode(code) {
    Atomics.store(this.int32View, Offset.ERROR_CODE / 4, code);
  }
  getFlags() {
    return Atomics.load(this.int32View, Offset.FLAGS / 4);
  }
  setFlags(flags) {
    Atomics.store(this.int32View, Offset.FLAGS / 4, flags);
  }
  getMode() {
    return Atomics.load(this.int32View, Offset.MODE / 4);
  }
  setMode(mode) {
    Atomics.store(this.int32View, Offset.MODE / 4, mode);
  }
  getPath() {
    const length = this.getPathLength();
    const bytes = this.uint8View.slice(
      Offset.PATH_BUFFER,
      Offset.PATH_BUFFER + length
    );
    return new TextDecoder().decode(bytes);
  }
  setPath(path) {
    const encoded = new TextEncoder().encode(path);
    if (encoded.length > Size.PATH_BUFFER) {
      throw new Error(`Path too long: ${encoded.length} > ${Size.PATH_BUFFER}`);
    }
    this.uint8View.set(encoded, Offset.PATH_BUFFER);
    this.setPathLength(encoded.length);
  }
  getData() {
    const length = this.getDataLength();
    return this.uint8View.slice(
      Offset.DATA_BUFFER,
      Offset.DATA_BUFFER + length
    );
  }
  setData(data) {
    if (data.length > Size.DATA_BUFFER) {
      throw new Error(`Data too large: ${data.length} > ${Size.DATA_BUFFER}`);
    }
    this.uint8View.set(data, Offset.DATA_BUFFER);
    this.setDataLength(data.length);
  }
  getDataAsString() {
    const data = this.getData();
    return new TextDecoder().decode(data);
  }
  setDataFromString(str) {
    const encoded = new TextEncoder().encode(str);
    this.setData(encoded);
  }
  getResult() {
    const length = this.getResultLength();
    return this.uint8View.slice(
      Offset.DATA_BUFFER,
      Offset.DATA_BUFFER + length
    );
  }
  setResult(data) {
    if (data.length > Size.DATA_BUFFER) {
      throw new Error(`Result too large: ${data.length} > ${Size.DATA_BUFFER}`);
    }
    this.uint8View.set(data, Offset.DATA_BUFFER);
    this.setResultLength(data.length);
  }
  getResultAsString() {
    const result = this.getResult();
    return new TextDecoder().decode(result);
  }
  setResultFromString(str) {
    const encoded = new TextEncoder().encode(str);
    this.setResult(encoded);
  }
  encodeStat(stat) {
    this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_FILE] = stat.isFile
      ? 1
      : 0;
    this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_DIRECTORY] =
      stat.isDirectory ? 1 : 0;
    this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_SYMLINK] =
      stat.isSymbolicLink ? 1 : 0;
    this.dataView.setInt32(
      Offset.DATA_BUFFER + StatLayout.MODE,
      stat.mode,
      true
    );
    const size = Math.min(stat.size, Number.MAX_SAFE_INTEGER);
    this.dataView.setFloat64(Offset.DATA_BUFFER + StatLayout.SIZE, size, true);
    this.dataView.setFloat64(
      Offset.DATA_BUFFER + StatLayout.MTIME,
      stat.mtime.getTime(),
      true
    );
    this.setResultLength(StatLayout.TOTAL);
  }
  decodeStat() {
    return {
      isFile: this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_FILE] === 1,
      isDirectory:
        this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_DIRECTORY] === 1,
      isSymbolicLink:
        this.uint8View[Offset.DATA_BUFFER + StatLayout.IS_SYMLINK] === 1,
      mode: this.dataView.getInt32(Offset.DATA_BUFFER + StatLayout.MODE, true),
      size: this.dataView.getFloat64(
        Offset.DATA_BUFFER + StatLayout.SIZE,
        true
      ),
      mtime: new Date(
        this.dataView.getFloat64(Offset.DATA_BUFFER + StatLayout.MTIME, true)
      )
    };
  }
  waitForReady(timeout) {
    return Atomics.wait(
      this.int32View,
      Offset.STATUS / 4,
      Status.PENDING,
      timeout
    );
  }
  waitForReadyAsync(timeout) {
    return Atomics.waitAsync(
      this.int32View,
      Offset.STATUS / 4,
      Status.PENDING,
      timeout
    );
  }
  /**
   * Wait for status to become READY.
   * Returns immediately if status is already READY, or waits until it changes.
   */
  async waitUntilReady(timeout) {
    const startTime = Date.now();
    while (true) {
      const status = this.getStatus();
      if (status === Status.READY) {
        return true;
      }
      const elapsed = Date.now() - startTime;
      if (elapsed >= timeout) {
        return false;
      }
      const remainingMs = timeout - elapsed;
      const result = Atomics.waitAsync(
        this.int32View,
        Offset.STATUS / 4,
        status,
        remainingMs
      );
      if (result.async) {
        const waitResult = await result.value;
        if (waitResult === "timed-out") {
          return false;
        }
      }
    }
  }
  waitForResult(timeout) {
    return Atomics.wait(
      this.int32View,
      Offset.STATUS / 4,
      Status.READY,
      timeout
    );
  }
  notify() {
    return Atomics.notify(this.int32View, Offset.STATUS / 4);
  }
  reset() {
    this.setOpCode(OpCode.NOOP);
    this.setStatus(Status.PENDING);
    this.setPathLength(0);
    this.setDataLength(0);
    this.setResultLength(0);
    this.setErrorCode(ErrorCode.NONE);
    this.setFlags(Flags.NONE);
    this.setMode(0);
  }
};

// src/commands/python3/sync-fs-backend.ts
var SyncFsBackend = class {
  protocol;
  constructor(sharedBuffer) {
    this.protocol = new ProtocolBuffer(sharedBuffer);
  }
  execSync(opCode, path, data, flags = 0, mode = 0) {
    this.protocol.reset();
    this.protocol.setOpCode(opCode);
    this.protocol.setPath(path);
    this.protocol.setFlags(flags);
    this.protocol.setMode(mode);
    if (data) {
      this.protocol.setData(data);
    }
    this.protocol.setStatus(Status.READY);
    this.protocol.notify();
    const waitResult = this.protocol.waitForResult(5e3);
    if (waitResult === "timed-out") {
      return { success: false, error: "Operation timed out" };
    }
    const status = this.protocol.getStatus();
    if (status === Status.SUCCESS) {
      return { success: true, result: this.protocol.getResult() };
    }
    return {
      success: false,
      error:
        this.protocol.getResultAsString() ||
        `Error code: ${this.protocol.getErrorCode()}`
    };
  }
  readFile(path) {
    const result = this.execSync(OpCode.READ_FILE, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to read file");
    }
    return result.result ?? new Uint8Array(0);
  }
  writeFile(path, data) {
    const result = this.execSync(OpCode.WRITE_FILE, path, data);
    if (!result.success) {
      throw new Error(result.error || "Failed to write file");
    }
  }
  stat(path) {
    const result = this.execSync(OpCode.STAT, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to stat");
    }
    return this.protocol.decodeStat();
  }
  lstat(path) {
    const result = this.execSync(OpCode.LSTAT, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to lstat");
    }
    return this.protocol.decodeStat();
  }
  readdir(path) {
    const result = this.execSync(OpCode.READDIR, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to readdir");
    }
    return JSON.parse(this.protocol.getResultAsString());
  }
  mkdir(path, recursive = false) {
    const flags = recursive ? Flags.MKDIR_RECURSIVE : 0;
    const result = this.execSync(OpCode.MKDIR, path, void 0, flags);
    if (!result.success) {
      throw new Error(result.error || "Failed to mkdir");
    }
  }
  rm(path, recursive = false, force = false) {
    let flags = 0;
    if (recursive) flags |= Flags.RECURSIVE;
    if (force) flags |= Flags.FORCE;
    const result = this.execSync(OpCode.RM, path, void 0, flags);
    if (!result.success) {
      throw new Error(result.error || "Failed to rm");
    }
  }
  exists(path) {
    const result = this.execSync(OpCode.EXISTS, path);
    if (!result.success) {
      return false;
    }
    return result.result?.[0] === 1;
  }
  appendFile(path, data) {
    const result = this.execSync(OpCode.APPEND_FILE, path, data);
    if (!result.success) {
      throw new Error(result.error || "Failed to append file");
    }
  }
  symlink(target, linkPath) {
    const targetData = new TextEncoder().encode(target);
    const result = this.execSync(OpCode.SYMLINK, linkPath, targetData);
    if (!result.success) {
      throw new Error(result.error || "Failed to symlink");
    }
  }
  readlink(path) {
    const result = this.execSync(OpCode.READLINK, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to readlink");
    }
    return this.protocol.getResultAsString();
  }
  chmod(path, mode) {
    const result = this.execSync(OpCode.CHMOD, path, void 0, 0, mode);
    if (!result.success) {
      throw new Error(result.error || "Failed to chmod");
    }
  }
  realpath(path) {
    const result = this.execSync(OpCode.REALPATH, path);
    if (!result.success) {
      throw new Error(result.error || "Failed to realpath");
    }
    return this.protocol.getResultAsString();
  }
  writeStdout(data) {
    const encoded = new TextEncoder().encode(data);
    this.execSync(OpCode.WRITE_STDOUT, "", encoded);
  }
  writeStderr(data) {
    const encoded = new TextEncoder().encode(data);
    this.execSync(OpCode.WRITE_STDERR, "", encoded);
  }
  exit(code) {
    this.execSync(OpCode.EXIT, "", void 0, code);
  }
  /**
   * Make an HTTP request through the main thread's secureFetch.
   * Returns the response as a parsed object.
   */
  httpRequest(url, options) {
    const requestData = options
      ? new TextEncoder().encode(JSON.stringify(options))
      : void 0;
    const result = this.execSync(OpCode.HTTP_REQUEST, url, requestData);
    if (!result.success) {
      throw new Error(result.error || "HTTP request failed");
    }
    const responseJson = new TextDecoder().decode(result.result);
    return JSON.parse(responseJson);
  }
};

// src/commands/python3/worker.ts
var pyodideInstance = null;
var pyodideLoading = null;
var require2 = createRequire(import.meta.url);
var pyodideIndexURL = `${dirname(require2.resolve("pyodide/pyodide.mjs"))}/`;
async function getPyodide() {
  if (pyodideInstance) {
    return pyodideInstance;
  }
  if (pyodideLoading) {
    return pyodideLoading;
  }
  pyodideLoading = loadPyodide({ indexURL: pyodideIndexURL });
  pyodideInstance = await pyodideLoading;
  return pyodideInstance;
}
function createHOSTFS(backend, FS, PATH) {
  const ERRNO_CODES = {
    EPERM: 63,
    ENOENT: 44,
    EIO: 29,
    EBADF: 8,
    EAGAIN: 6,
    EACCES: 2,
    EBUSY: 10,
    EEXIST: 20,
    ENOTDIR: 54,
    EISDIR: 31,
    EINVAL: 28,
    EMFILE: 33,
    ENOSPC: 51,
    ESPIPE: 70,
    EROFS: 69,
    ENOTEMPTY: 55,
    ENOSYS: 52,
    ENOTSUP: 138,
    ENODATA: 42
  };
  function realPath(node) {
    const parts = [];
    while (node.parent !== node) {
      parts.push(node.name);
      node = node.parent;
    }
    parts.push(node.mount.opts.root);
    parts.reverse();
    return PATH.join(...parts);
  }
  function tryFSOperation(f) {
    try {
      return f();
    } catch (e) {
      const msg =
        e?.message?.toLowerCase() ||
        (typeof e === "string" ? e.toLowerCase() : "");
      let code = ERRNO_CODES.EIO;
      if (msg.includes("no such file") || msg.includes("not found")) {
        code = ERRNO_CODES.ENOENT;
      } else if (msg.includes("is a directory")) {
        code = ERRNO_CODES.EISDIR;
      } else if (msg.includes("not a directory")) {
        code = ERRNO_CODES.ENOTDIR;
      } else if (msg.includes("already exists")) {
        code = ERRNO_CODES.EEXIST;
      } else if (msg.includes("permission")) {
        code = ERRNO_CODES.EACCES;
      } else if (msg.includes("not empty")) {
        code = ERRNO_CODES.ENOTEMPTY;
      }
      throw new FS.ErrnoError(code);
    }
  }
  function getMode(path) {
    return tryFSOperation(() => {
      const stat = backend.stat(path);
      let mode = stat.mode & 511;
      if (stat.isDirectory) {
        mode |= 16384;
      } else if (stat.isSymbolicLink) {
        mode |= 40960;
      } else {
        mode |= 32768;
      }
      return mode;
    });
  }
  const HOSTFS = {
    mount(_mount) {
      return HOSTFS.createNode(null, "/", 16877, 0);
    },
    createNode(parent, name, mode, dev) {
      if (!FS.isDir(mode) && !FS.isFile(mode) && !FS.isLink(mode)) {
        throw new FS.ErrnoError(ERRNO_CODES.EINVAL);
      }
      const node = FS.createNode(parent, name, mode, dev);
      node.node_ops = HOSTFS.node_ops;
      node.stream_ops = HOSTFS.stream_ops;
      return node;
    },
    node_ops: {
      getattr(node) {
        const path = realPath(node);
        return tryFSOperation(() => {
          const stat = backend.stat(path);
          let mode = stat.mode & 511;
          if (stat.isDirectory) {
            mode |= 16384;
          } else if (stat.isSymbolicLink) {
            mode |= 40960;
          } else {
            mode |= 32768;
          }
          return {
            dev: 1,
            ino: node.id,
            mode,
            nlink: 1,
            uid: 0,
            gid: 0,
            rdev: 0,
            size: stat.size,
            atime: stat.mtime,
            mtime: stat.mtime,
            ctime: stat.mtime,
            blksize: 4096,
            blocks: Math.ceil(stat.size / 512)
          };
        });
      },
      setattr(node, attr) {
        const path = realPath(node);
        const mode = attr.mode;
        if (mode !== void 0) {
          tryFSOperation(() => backend.chmod(path, mode));
          node.mode = mode;
        }
        if (attr.size !== void 0) {
          tryFSOperation(() => {
            const content = backend.readFile(path);
            const newContent = content.slice(0, attr.size);
            backend.writeFile(path, newContent);
          });
        }
      },
      lookup(parent, name) {
        const path = PATH.join2(realPath(parent), name);
        const mode = getMode(path);
        return HOSTFS.createNode(parent, name, mode);
      },
      mknod(parent, name, mode, _dev) {
        const node = HOSTFS.createNode(parent, name, mode, _dev);
        const path = realPath(node);
        tryFSOperation(() => {
          if (FS.isDir(node.mode)) {
            backend.mkdir(path, false);
          } else {
            backend.writeFile(path, new Uint8Array(0));
          }
        });
        return node;
      },
      rename(oldNode, newDir, newName) {
        const oldPath = realPath(oldNode);
        const newPath = PATH.join2(realPath(newDir), newName);
        tryFSOperation(() => {
          const content = backend.readFile(oldPath);
          backend.writeFile(newPath, content);
          backend.rm(oldPath, false, false);
        });
        oldNode.name = newName;
      },
      unlink(parent, name) {
        const path = PATH.join2(realPath(parent), name);
        tryFSOperation(() => backend.rm(path, false, false));
      },
      rmdir(parent, name) {
        const path = PATH.join2(realPath(parent), name);
        tryFSOperation(() => backend.rm(path, false, false));
      },
      readdir(node) {
        const path = realPath(node);
        return tryFSOperation(() => backend.readdir(path));
      },
      symlink(parent, newName, oldPath) {
        const newPath = PATH.join2(realPath(parent), newName);
        tryFSOperation(() => backend.symlink(oldPath, newPath));
      },
      readlink(node) {
        const path = realPath(node);
        return tryFSOperation(() => backend.readlink(path));
      }
    },
    stream_ops: {
      open(stream) {
        const path = realPath(stream.node);
        const flags = stream.flags;
        const O_WRONLY = 1;
        const O_RDWR = 2;
        const O_CREAT = 64;
        const O_TRUNC = 512;
        const O_APPEND = 1024;
        const accessMode = flags & 3;
        const isWrite = accessMode === O_WRONLY || accessMode === O_RDWR;
        const isCreate = (flags & O_CREAT) !== 0;
        const isTruncate = (flags & O_TRUNC) !== 0;
        const isAppend = (flags & O_APPEND) !== 0;
        if (FS.isDir(stream.node.mode)) {
          return;
        }
        let content;
        try {
          if (isTruncate && isWrite) {
            content = new Uint8Array(0);
          } else {
            content = backend.readFile(path);
          }
        } catch (_e) {
          if (isCreate && isWrite) {
            content = new Uint8Array(0);
          } else {
            throw new FS.ErrnoError(ERRNO_CODES.ENOENT);
          }
        }
        stream.hostContent = content;
        stream.hostModified = isTruncate && isWrite;
        stream.hostPath = path;
        if (isAppend) {
          stream.position = content.length;
        }
      },
      close(stream) {
        const hostPath = stream.hostPath;
        const hostContent = stream.hostContent;
        if (stream.hostModified && hostContent && hostPath) {
          tryFSOperation(() => backend.writeFile(hostPath, hostContent));
        }
        delete stream.hostContent;
        delete stream.hostModified;
        delete stream.hostPath;
      },
      read(stream, buffer, offset, length, position) {
        const content = stream.hostContent;
        if (!content) return 0;
        const size = content.length;
        if (position >= size) return 0;
        const bytesToRead = Math.min(length, size - position);
        buffer.set(content.subarray(position, position + bytesToRead), offset);
        return bytesToRead;
      },
      write(stream, buffer, offset, length, position) {
        let content = stream.hostContent || new Uint8Array(0);
        const newSize = Math.max(content.length, position + length);
        if (newSize > content.length) {
          const newContent = new Uint8Array(newSize);
          newContent.set(content);
          content = newContent;
          stream.hostContent = content;
        }
        content.set(buffer.subarray(offset, offset + length), position);
        stream.hostModified = true;
        return length;
      },
      llseek(stream, offset, whence) {
        const SEEK_CUR = 1;
        const SEEK_END = 2;
        let position = offset;
        if (whence === SEEK_CUR) {
          position += stream.position;
        } else if (whence === SEEK_END) {
          if (FS.isFile(stream.node.mode)) {
            const content = stream.hostContent;
            position += content ? content.length : 0;
          }
        }
        if (position < 0) {
          throw new FS.ErrnoError(ERRNO_CODES.EINVAL);
        }
        return position;
      }
    }
  };
  return HOSTFS;
}
async function runPython(input) {
  const backend = new SyncFsBackend(input.sharedBuffer);
  let pyodide;
  try {
    pyodide = await getPyodide();
  } catch (e) {
    return {
      success: false,
      error: `Failed to load Pyodide: ${e.message}`
    };
  }
  pyodide.setStdout({ batched: () => {} });
  pyodide.setStderr({ batched: () => {} });
  try {
    pyodide.runPython(`
import sys
if hasattr(sys.stdout, 'flush'):
    sys.stdout.flush()
if hasattr(sys.stderr, 'flush'):
    sys.stderr.flush()
`);
  } catch (_e) {}
  pyodide.setStdout({
    batched: (text) => {
      backend.writeStdout(`${text}
`);
    }
  });
  pyodide.setStderr({
    batched: (text) => {
      backend.writeStderr(`${text}
`);
    }
  });
  const FS = pyodide.FS;
  const PATH = pyodide.PATH;
  const HOSTFS = createHOSTFS(backend, FS, PATH);
  try {
    try {
      pyodide.runPython(`import os; os.chdir('/')`);
    } catch (_e) {}
    try {
      FS.mkdir("/host");
    } catch (_e) {}
    try {
      FS.unmount("/host");
    } catch (_e) {}
    FS.mount(HOSTFS, { root: "/" }, "/host");
  } catch (e) {
    return {
      success: false,
      error: `Failed to mount HOSTFS: ${e.message}`
    };
  }
  try {
    pyodide.runPython(`
import sys
if '_jb_http_bridge' in sys.modules:
    del sys.modules['_jb_http_bridge']
if 'jb_http' in sys.modules:
    del sys.modules['jb_http']
`);
  } catch (_e) {}
  pyodide.registerJsModule("_jb_http_bridge", {
    request: (url, method, headersJson, body) => {
      try {
        const headers = headersJson ? JSON.parse(headersJson) : void 0;
        const result = backend.httpRequest(url, {
          method: method || "GET",
          headers,
          body: body || void 0
        });
        return JSON.stringify(result);
      } catch (e) {
        return JSON.stringify({ error: e.message });
      }
    }
  });
  const envSetup = Object.entries(input.env)
    .map(([key, value]) => {
      return `os.environ[${JSON.stringify(key)}] = ${JSON.stringify(value)}`;
    })
    .join("\n");
  const argv0 = input.scriptPath || "python3";
  const argvList = [argv0, ...input.args]
    .map((arg) => JSON.stringify(arg))
    .join(", ");
  try {
    await pyodide.runPythonAsync(`
import os
import sys
import builtins
import json

${envSetup}

sys.argv = [${argvList}]

# Create jb_http module for HTTP requests
class _JbHttpResponse:
    """HTTP response object similar to requests.Response"""
    def __init__(self, data):
        self.status_code = data.get('status', 0)
        self.reason = data.get('statusText', '')
        # @banned-pattern-ignore: Python code, not JavaScript
        self.headers = data.get('headers', {})
        self.text = data.get('body', '')
        self.url = data.get('url', '')
        self._error = data.get('error')

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self._error:
            raise Exception(self._error)
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}: {self.reason}")

class _JbHttp:
    """HTTP client that bridges to just-bash's secureFetch"""
    def request(self, method, url, headers=None, data=None, json_data=None):
        # Import fresh each time to ensure we use the current bridge
        # (important when worker is reused with different SharedArrayBuffer)
        import _jb_http_bridge
        if json_data is not None:
            data = json.dumps(json_data)
            headers = headers or {}
            headers['Content-Type'] = 'application/json'
        # Serialize headers to JSON to avoid PyProxy issues when passing to JS
        headers_json = json.dumps(headers) if headers else None
        result_json = _jb_http_bridge.request(url, method, headers_json, data)
        result = json.loads(result_json)
        # Check for errors from the bridge (network not configured, URL not allowed, etc.)
        if 'error' in result and result.get('status') is None:
            raise Exception(result['error'])
        return _JbHttpResponse(result)

    def get(self, url, headers=None, **kwargs):
        return self.request('GET', url, headers=headers, **kwargs)

    def post(self, url, headers=None, data=None, json=None, **kwargs):
        return self.request('POST', url, headers=headers, data=data, json_data=json, **kwargs)

    def put(self, url, headers=None, data=None, json=None, **kwargs):
        return self.request('PUT', url, headers=headers, data=data, json_data=json, **kwargs)

    def delete(self, url, headers=None, **kwargs):
        return self.request('DELETE', url, headers=headers, **kwargs)

    def head(self, url, headers=None, **kwargs):
        return self.request('HEAD', url, headers=headers, **kwargs)

    def patch(self, url, headers=None, data=None, json=None, **kwargs):
        return self.request('PATCH', url, headers=headers, data=data, json_data=json, **kwargs)

# Register jb_http as an importable module
import types
jb_http = types.ModuleType('jb_http')
jb_http._client = _JbHttp()
jb_http.get = jb_http._client.get
jb_http.post = jb_http._client.post
jb_http.put = jb_http._client.put
jb_http.delete = jb_http._client.delete
jb_http.head = jb_http._client.head
jb_http.patch = jb_http._client.patch
jb_http.request = jb_http._client.request
jb_http.Response = _JbHttpResponse
sys.modules['jb_http'] = jb_http

# ============================================================
# SANDBOX SECURITY SETUP
# ============================================================
# Only apply sandbox restrictions once per Pyodide instance
if not hasattr(builtins, '_jb_sandbox_initialized'):
    builtins._jb_sandbox_initialized = True

    # ------------------------------------------------------------
    # 1. Block dangerous module imports (js, pyodide, pyodide_js, pyodide.ffi)
    # These allow sandbox escape via JavaScript execution
    # ------------------------------------------------------------
    _BLOCKED_MODULES = frozenset({'js', 'pyodide', 'pyodide_js', 'pyodide.ffi'})
    _BLOCKED_PREFIXES = ('js.', 'pyodide.', 'pyodide_js.')

    # Remove pre-loaded dangerous modules from sys.modules
    for _blocked_mod in list(sys.modules.keys()):
        if _blocked_mod in _BLOCKED_MODULES or any(_blocked_mod.startswith(p) for p in _BLOCKED_PREFIXES):
            del sys.modules[_blocked_mod]

    # Create a secure callable wrapper that hides introspection attributes
    # This prevents access to __closure__, __kwdefaults__, __globals__, etc.
    def _make_secure_import(orig_import, blocked, prefixes):
        """Create import function wrapped to block introspection."""
        def _inner(name, globals=None, locals=None, fromlist=(), level=0):
            if name in blocked or any(name.startswith(p) for p in prefixes):
                raise ImportError(f"Module '{name}' is blocked in this sandbox")
            return orig_import(name, globals, locals, fromlist, level)

        class _SecureImport:
            """Wrapper that hides function internals from introspection."""
            __slots__ = ()
            def __call__(self, name, globals=None, locals=None, fromlist=(), level=0):
                return _inner(name, globals, locals, fromlist, level)
            def __getattribute__(self, name):
                if name in ('__call__', '__class__'):
                    return object.__getattribute__(self, name)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            def __repr__(self):
                return '<built-in function __import__>'
        return _SecureImport()

    builtins.__import__ = _make_secure_import(builtins.__import__, _BLOCKED_MODULES, _BLOCKED_PREFIXES)
    del _BLOCKED_MODULES, _BLOCKED_PREFIXES, _make_secure_import

    # ------------------------------------------------------------
    # 2. Path redirection helper
    # ------------------------------------------------------------
    def _should_redirect(path):
        """Check if a path should be redirected to /host."""
        return (isinstance(path, str) and
                path.startswith('/') and
                not path.startswith('/lib') and
                not path.startswith('/proc') and
                not path.startswith('/host'))

    # ------------------------------------------------------------
    # 3. Secure wrapper factory for file operations
    # ------------------------------------------------------------
    # This creates callable wrappers that hide __closure__, __globals__, etc.
    def _make_secure_wrapper(func, name):
        """Wrap a function to block introspection attributes."""
        class _SecureWrapper:
            __slots__ = ()
            def __call__(self, *args, **kwargs):
                return func(*args, **kwargs)
            def __getattribute__(self, attr):
                if attr in ('__call__', '__class__'):
                    return object.__getattribute__(self, attr)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
            def __repr__(self):
                return f'<built-in function {name}>'
        return _SecureWrapper()

    # ------------------------------------------------------------
    # 4. Redirect file operations to /host (with secure wrappers)
    # ------------------------------------------------------------
    # builtins.open
    _orig_open = builtins.open
    def _redir_open(path, mode='r', *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_open(path, mode, *args, **kwargs)
    builtins.open = _make_secure_wrapper(_redir_open, 'open')

    # os.listdir
    _orig_listdir = os.listdir
    def _redir_listdir(path='.'):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_listdir(path)
    os.listdir = _make_secure_wrapper(_redir_listdir, 'listdir')

    # os.path.exists
    _orig_exists = os.path.exists
    def _redir_exists(path):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_exists(path)
    os.path.exists = _make_secure_wrapper(_redir_exists, 'exists')

    # os.path.isfile
    _orig_isfile = os.path.isfile
    def _redir_isfile(path):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_isfile(path)
    os.path.isfile = _make_secure_wrapper(_redir_isfile, 'isfile')

    # os.path.isdir
    _orig_isdir = os.path.isdir
    def _redir_isdir(path):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_isdir(path)
    os.path.isdir = _make_secure_wrapper(_redir_isdir, 'isdir')

    # os.stat
    _orig_stat = os.stat
    def _redir_stat(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_stat(path, *args, **kwargs)
    os.stat = _make_secure_wrapper(_redir_stat, 'stat')

    # os.mkdir
    _orig_mkdir = os.mkdir
    def _redir_mkdir(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_mkdir(path, *args, **kwargs)
    os.mkdir = _make_secure_wrapper(_redir_mkdir, 'mkdir')

    # os.makedirs
    _orig_makedirs = os.makedirs
    def _redir_makedirs(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_makedirs(path, *args, **kwargs)
    os.makedirs = _make_secure_wrapper(_redir_makedirs, 'makedirs')

    # os.remove
    _orig_remove = os.remove
    def _redir_remove(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_remove(path, *args, **kwargs)
    os.remove = _make_secure_wrapper(_redir_remove, 'remove')

    # os.rmdir
    _orig_rmdir = os.rmdir
    def _redir_rmdir(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_rmdir(path, *args, **kwargs)
    os.rmdir = _make_secure_wrapper(_redir_rmdir, 'rmdir')

    # os.getcwd - strip /host prefix
    _orig_getcwd = os.getcwd
    def _redir_getcwd():
        cwd = _orig_getcwd()
        if cwd.startswith('/host'):
            return cwd[5:]  # Strip '/host' prefix
        return cwd
    os.getcwd = _make_secure_wrapper(_redir_getcwd, 'getcwd')

    # os.chdir
    _orig_chdir = os.chdir
    def _redir_chdir(path):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_chdir(path)
    os.chdir = _make_secure_wrapper(_redir_chdir, 'chdir')

    # ------------------------------------------------------------
    # 5. Additional file operations (glob, walk, scandir, io.open)
    # ------------------------------------------------------------
    import glob as _glob_module

    _orig_glob = _glob_module.glob
    def _redir_glob(pathname, *args, **kwargs):
        if _should_redirect(pathname):
            pathname = '/host' + pathname
        return _orig_glob(pathname, *args, **kwargs)
    _glob_module.glob = _make_secure_wrapper(_redir_glob, 'glob')

    _orig_iglob = _glob_module.iglob
    def _redir_iglob(pathname, *args, **kwargs):
        if _should_redirect(pathname):
            pathname = '/host' + pathname
        return _orig_iglob(pathname, *args, **kwargs)
    _glob_module.iglob = _make_secure_wrapper(_redir_iglob, 'iglob')

    # os.walk (generator - needs special handling)
    _orig_walk = os.walk
    def _redir_walk(top, *args, **kwargs):
        redirected = False
        if _should_redirect(top):
            top = '/host' + top
            redirected = True
        for dirpath, dirnames, filenames in _orig_walk(top, *args, **kwargs):
            if redirected and dirpath.startswith('/host'):
                dirpath = dirpath[5:] if len(dirpath) > 5 else '/'
            yield dirpath, dirnames, filenames
    os.walk = _make_secure_wrapper(_redir_walk, 'walk')

    # os.scandir
    _orig_scandir = os.scandir
    def _redir_scandir(path='.'):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_scandir(path)
    os.scandir = _make_secure_wrapper(_redir_scandir, 'scandir')

    # io.open (same secure wrapper as builtins.open)
    import io as _io_module
    _io_module.open = builtins.open

    # ------------------------------------------------------------
    # 6. shutil file operations
    # ------------------------------------------------------------
    import shutil as _shutil_module

    # shutil.copy(src, dst)
    _orig_shutil_copy = _shutil_module.copy
    def _redir_shutil_copy(src, dst, *args, **kwargs):
        if _should_redirect(src):
            src = '/host' + src
        if _should_redirect(dst):
            dst = '/host' + dst
        return _orig_shutil_copy(src, dst, *args, **kwargs)
    _shutil_module.copy = _make_secure_wrapper(_redir_shutil_copy, 'copy')

    # shutil.copy2(src, dst)
    _orig_shutil_copy2 = _shutil_module.copy2
    def _redir_shutil_copy2(src, dst, *args, **kwargs):
        if _should_redirect(src):
            src = '/host' + src
        if _should_redirect(dst):
            dst = '/host' + dst
        return _orig_shutil_copy2(src, dst, *args, **kwargs)
    _shutil_module.copy2 = _make_secure_wrapper(_redir_shutil_copy2, 'copy2')

    # shutil.copyfile(src, dst)
    _orig_shutil_copyfile = _shutil_module.copyfile
    def _redir_shutil_copyfile(src, dst, *args, **kwargs):
        if _should_redirect(src):
            src = '/host' + src
        if _should_redirect(dst):
            dst = '/host' + dst
        return _orig_shutil_copyfile(src, dst, *args, **kwargs)
    _shutil_module.copyfile = _make_secure_wrapper(_redir_shutil_copyfile, 'copyfile')

    # shutil.copytree(src, dst)
    _orig_shutil_copytree = _shutil_module.copytree
    def _redir_shutil_copytree(src, dst, *args, **kwargs):
        if _should_redirect(src):
            src = '/host' + src
        if _should_redirect(dst):
            dst = '/host' + dst
        return _orig_shutil_copytree(src, dst, *args, **kwargs)
    _shutil_module.copytree = _make_secure_wrapper(_redir_shutil_copytree, 'copytree')

    # shutil.move(src, dst)
    _orig_shutil_move = _shutil_module.move
    def _redir_shutil_move(src, dst, *args, **kwargs):
        if _should_redirect(src):
            src = '/host' + src
        if _should_redirect(dst):
            dst = '/host' + dst
        return _orig_shutil_move(src, dst, *args, **kwargs)
    _shutil_module.move = _make_secure_wrapper(_redir_shutil_move, 'move')

    # shutil.rmtree(path)
    _orig_shutil_rmtree = _shutil_module.rmtree
    def _redir_shutil_rmtree(path, *args, **kwargs):
        if _should_redirect(path):
            path = '/host' + path
        return _orig_shutil_rmtree(path, *args, **kwargs)
    _shutil_module.rmtree = _make_secure_wrapper(_redir_shutil_rmtree, 'rmtree')

    # ------------------------------------------------------------
    # 7. pathlib.Path - redirect path resolution
    # ------------------------------------------------------------
    from pathlib import Path, PurePosixPath

    def _redirect_path(p):
        """Convert a Path to redirect /absolute paths to /host."""
        s = str(p)
        if _should_redirect(s):
            return Path('/host' + s)
        return p

    # Helper to create method wrappers for Path
    def _wrap_path_method(orig_method, name):
        def wrapper(self, *args, **kwargs):
            redirected = _redirect_path(self)
            return getattr(redirected, '_orig_' + name)(*args, **kwargs)
        return wrapper

    # Store original methods with _orig_ prefix, then replace with redirecting versions
    # Path.stat()
    Path._orig_stat = Path.stat
    def _path_stat(self, *args, **kwargs):
        return _redirect_path(self)._orig_stat(*args, **kwargs)
    Path.stat = _path_stat

    # Path.exists()
    Path._orig_exists = Path.exists
    def _path_exists(self):
        return _redirect_path(self)._orig_exists()
    Path.exists = _path_exists

    # Path.is_file()
    Path._orig_is_file = Path.is_file
    def _path_is_file(self):
        return _redirect_path(self)._orig_is_file()
    Path.is_file = _path_is_file

    # Path.is_dir()
    Path._orig_is_dir = Path.is_dir
    def _path_is_dir(self):
        return _redirect_path(self)._orig_is_dir()
    Path.is_dir = _path_is_dir

    # Path.open()
    Path._orig_open = Path.open
    def _path_open(self, *args, **kwargs):
        return _redirect_path(self)._orig_open(*args, **kwargs)
    Path.open = _path_open

    # Path.read_text()
    Path._orig_read_text = Path.read_text
    def _path_read_text(self, *args, **kwargs):
        return _redirect_path(self)._orig_read_text(*args, **kwargs)
    Path.read_text = _path_read_text

    # Path.read_bytes()
    Path._orig_read_bytes = Path.read_bytes
    def _path_read_bytes(self):
        return _redirect_path(self)._orig_read_bytes()
    Path.read_bytes = _path_read_bytes

    # Path.write_text()
    Path._orig_write_text = Path.write_text
    def _path_write_text(self, *args, **kwargs):
        return _redirect_path(self)._orig_write_text(*args, **kwargs)
    Path.write_text = _path_write_text

    # Path.write_bytes()
    Path._orig_write_bytes = Path.write_bytes
    def _path_write_bytes(self, data):
        return _redirect_path(self)._orig_write_bytes(data)
    Path.write_bytes = _path_write_bytes

    # Path.mkdir()
    Path._orig_mkdir = Path.mkdir
    def _path_mkdir(self, *args, **kwargs):
        return _redirect_path(self)._orig_mkdir(*args, **kwargs)
    Path.mkdir = _path_mkdir

    # Path.rmdir()
    Path._orig_rmdir = Path.rmdir
    def _path_rmdir(self):
        return _redirect_path(self)._orig_rmdir()
    Path.rmdir = _path_rmdir

    # Path.unlink()
    Path._orig_unlink = Path.unlink
    def _path_unlink(self, *args, **kwargs):
        return _redirect_path(self)._orig_unlink(*args, **kwargs)
    Path.unlink = _path_unlink

    # Path.iterdir()
    Path._orig_iterdir = Path.iterdir
    def _path_iterdir(self):
        redirected = _redirect_path(self)
        for p in redirected._orig_iterdir():
            # Strip /host prefix from results
            s = str(p)
            if s.startswith('/host'):
                yield Path(s[5:])
            else:
                yield p
    Path.iterdir = _path_iterdir

    # Path.glob()
    Path._orig_glob = Path.glob
    def _path_glob(self, pattern):
        redirected = _redirect_path(self)
        for p in redirected._orig_glob(pattern):
            s = str(p)
            if s.startswith('/host'):
                yield Path(s[5:])
            else:
                yield p
    Path.glob = _path_glob

    # Path.rglob()
    Path._orig_rglob = Path.rglob
    def _path_rglob(self, pattern):
        redirected = _redirect_path(self)
        for p in redirected._orig_rglob(pattern):
            s = str(p)
            if s.startswith('/host'):
                yield Path(s[5:])
            else:
                yield p
    Path.rglob = _path_rglob

# Set cwd to host mount
os.chdir('/host' + ${JSON.stringify(input.cwd)})
`);
  } catch (e) {
    return {
      success: false,
      error: `Failed to set up environment: ${e.message}`
    };
  }
  try {
    const wrappedCode = `
import sys
_jb_exit_code = 0
try:
${input.pythonCode
  .split("\n")
  .map((line) => `    ${line}`)
  .join("\n")}
except SystemExit as e:
    _jb_exit_code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
`;
    await pyodide.runPythonAsync(wrappedCode);
    const exitCode = pyodide.globals.get("_jb_exit_code");
    backend.exit(exitCode);
    return { success: true };
  } catch (e) {
    const error = e;
    backend.writeStderr(`${error.message}
`);
    backend.exit(1);
    return { success: true };
  }
}
var defense = null;
async function initializeWithDefense() {
  await getPyodide();
  defense = new WorkerDefenseInDepth({
    excludeViolationTypes: [
      "proxy",
      "setImmediate",
      // 3. SharedArrayBuffer/Atomics: Used by sync-fs-backend.ts for synchronous
      //    filesystem communication between Pyodide's WASM thread and the main thread.
      //    Without this, Pyodide cannot perform synchronous file I/O operations.
      "shared_array_buffer",
      "atomics"
    ],
    onViolation: (v) => {
      parentPort?.postMessage({ type: "security-violation", violation: v });
    }
  });
}
if (parentPort) {
  if (workerData) {
    initializeWithDefense()
      .then(() => runPython(workerData))
      .then((result) => {
        result.defenseStats = defense?.getStats();
        parentPort?.postMessage(result);
      })
      .catch((e) => {
        parentPort?.postMessage({
          success: false,
          error: e.message,
          defenseStats: defense?.getStats()
        });
      });
  }
  parentPort.on("message", async (input) => {
    try {
      if (!defense) {
        await initializeWithDefense();
      }
      const result = await runPython(input);
      result.defenseStats = defense?.getStats();
      parentPort?.postMessage(result);
    } catch (e) {
      parentPort?.postMessage({
        success: false,
        error: e.message,
        defenseStats: defense?.getStats()
      });
    }
  });
}
