// libkitjl — a flat `extern "C"` shim that hosts the NVIDIA Kit runtime
// IN-PROCESS for OmniverseKitMakie's `KitScreen` (the in-process transport,
// as opposed to the default headless `kit` subprocess).
//
// This is the ONLY C++ in the repo besides ovrtx.  It keeps every carb/omni
// C++ type behind an opaque `KitJlApp*` handle so Julia never touches a vtable
// or a carb interface struct.  It is a direct C++ port of `kit/kit_app.py`
// (the official in-process bootstrap): acquire the Carbonite Framework, load
// the `omni.kit.app` plugin, acquire `omni::kit::IApp`, and drive
// startup/update/shutdown.  Extension/setting configuration is handed to
// `IApp::startup` as the SAME argv vector the subprocess launch uses (the
// `kit` executable is a thin bootstrap that forwards argv to the app's
// extension/settings parser — so `--enable …`, `--ext-folder …`, `--/… `
// entries reproduce the subprocess launch exactly).
//
// Client-init pattern (grounded in kit/dev/include):
//   * `OMNI_APP_GLOBALS(...)` at global scope defines the required carb/omni
//     client globals (framework, logging, profiler, assert, l10n,
//     crashreporter, default ONI channel) — exactly once per app.
//   * `carb::acquireFrameworkAndRegisterBuiltins()` == `OMNI_CORE_INIT`'s
//     `ScopedOmniCore` == the Python `carb.get_framework()` client init.  We
//     deliberately do NOT call `carb::startupFramework()` (OMNI_CORE_INIT's
//     `ScopedFrameworkStartup`): the low-level kit_app.py path lets
//     `IApp::startup` handle config, avoiding startupFramework's default
//     config-file discovery.
//
// Every entry point try/catches into a thread-local last-error string; no C++
// exception ever crosses the ABI.  Mirrors LibOVRTX's check/last-error idiom.

#include <omni/core/OmniInit.h>       // OMNI_APP_GLOBALS, OMNI_CORE_*, acquireFrameworkAndRegisterBuiltins,
                                      // carb::getFramework, carb::loadPluginsFromPattern (via StartupUtils.h)
#include <omni/kit/IApp.h>            // omni::kit::IApp, IAppScripting, AppDesc
#include <carb/Framework.h>          // carbGetSdkVersion
#include <carb/settings/ISettings.h> // carb::settings::ISettings

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

// Carbonite client globals — required exactly once, at global scope, in one
// translation unit of a carb application (see carb/ClientUtils.h).
OMNI_APP_GLOBALS("libkitjl", "libkitjl in-process Kit host")

namespace
{
// Thread-local last-error, copied out by kitjl_last_error (LibOVRTX idiom).
thread_local std::string g_lastError;

void setError(const char* where, const char* what)
{
    g_lastError = std::string(where) + ": " + (what ? what : "(unknown)");
}
} // namespace

// Opaque handle: keeps all carb/omni C++ types out of Julia.
struct KitJlApp
{
    carb::Framework* framework = nullptr;
    omni::kit::IApp* app = nullptr;
    carb::settings::ISettings* settings = nullptr;
    omni::kit::IAppScripting* scripting = nullptr;
    bool started = false;
};

namespace
{
// Lazily acquire ISettings (carb.settings.plugin is loaded by IApp::startup,
// so this is null until after startup; acquire on first use).
carb::settings::ISettings* settingsOf(KitJlApp* h)
{
    if (h && !h->settings && h->framework)
    {
        h->settings = h->framework->tryAcquireInterface<carb::settings::ISettings>();
    }
    return h ? h->settings : nullptr;
}
} // namespace

extern "C"
{

    const char* kitjl_last_error(void)
    {
        return g_lastError.c_str();
    }

    // carbGetSdkVersion() is a free exported C function that returns the carb
    // SDK version string WITHOUT starting the framework or touching the GPU —
    // the pure-tier smoke test.
    const char* kitjl_sdk_version(void)
    {
        try
        {
            const char* v = carb::carbGetSdkVersion();
            return v ? v : "";
        }
        catch (...)
        {
            setError("kitjl_sdk_version", "exception");
            return "";
        }
    }

    KitJlApp* kitjl_startup(int argc, const char* const* argv)
    {
        try
        {
            g_lastError.clear();

            const char* appPath = std::getenv("CARB_APP_PATH");
            if (!appPath || !*appPath)
            {
                setError("kitjl_startup", "CARB_APP_PATH not set");
                return nullptr;
            }
            std::string pluginDir = std::string(appPath) + "/kernel/plugins";
            const char* searchPaths[] = { pluginDir.c_str() };

            // 1. Acquire framework + register builtins + OMNI_CORE_START.
            //    (mirrors ScopedOmniCore / the Python `carb.get_framework()`).
            if (!carb::getFramework())
            {
                carb::acquireFrameworkAndRegisterBuiltins();
            }
            carb::Framework* f = carb::getFramework();
            if (!f)
            {
                setError("kitjl_startup", "acquireFramework returned null");
                return nullptr;
            }

            // 1b. FULL framework startup from argv — loads the base carb plugins
            //     (settings/dictionary/tokens + serializer), reads config, wires
            //     logging, and applies the `--/…` overrides.  This is what the
            //     C++ `kit` binary does (OMNI_CORE_INIT(argc, argv)); the lighter
            //     kit_app.py path (no startupFramework) DEADLOCKS when Kit is
            //     co-hosted in the Julia process: omni.usd_resolver spins in its
            //     Python-import dlopen without tokens/config initialized.  The
            //     plugin search path MUST be Kit's kernel/plugins — the host
            //     executable here is `julia`, not `kit`, so the default
            //     exe-dir search would never find carb.settings.plugin.
            {
                carb::StartupFrameworkDesc sd = carb::StartupFrameworkDesc::getDefault();
                sd.argv = const_cast<char**>(argv);
                sd.argc = argc;
                sd.appNameOverride = "kit";
                sd.appPathOverride = appPath;
                sd.initialPluginsSearchPaths = searchPaths;
                sd.initialPluginsSearchPathCount = 1;
                sd.disableCrashReporter = true; // belt (+ Julia SignalGuard around this call)
                carb::startupFramework(sd);
            }

            // 2. Load the omni.kit.app plugin from <CARB_APP_PATH>/kernel/plugins
            //    (absolute path; kit_app.py uses ${CARB_APP_PATH}/kernel/plugins).
            carb::loadPluginsFromPattern("omni.kit.app.plugin", searchPaths, 1);

            // 3. Acquire IApp.
            omni::kit::IApp* app = f->acquireInterface<omni::kit::IApp>();
            if (!app)
            {
                setError("kitjl_startup", "acquireInterface<omni::kit::IApp> returned null");
                return nullptr;
            }

            // 4. startup(AppDesc): argv is forwarded to the extension/settings
            //    parser — the SAME argv the subprocess launch uses reproduces it.
            omni::kit::AppDesc desc{};
            desc.carbAppName = "kit";
            desc.carbAppPath = appPath;
            desc.argc = argc;
            desc.argv = const_cast<char**>(argv); // IApp::startup does not mutate argv
            app->startup(desc);

            // 5. Warmup updates so the viewport/renderer come up (matches
            //    kit_server.py's pre-ready updates).
            for (int i = 0; i < 3; ++i)
            {
                app->update();
            }

            KitJlApp* h = new (std::nothrow) KitJlApp();
            if (!h)
            {
                setError("kitjl_startup", "handle allocation failed");
                return nullptr;
            }
            h->framework = f;
            h->app = app;
            h->settings = f->tryAcquireInterface<carb::settings::ISettings>();
            h->scripting = app->getPythonScripting();
            h->started = true;
            return h;
        }
        catch (const std::exception& e)
        {
            setError("kitjl_startup", e.what());
            return nullptr;
        }
        catch (...)
        {
            setError("kitjl_startup", "unknown C++ exception");
            return nullptr;
        }
    }

    void kitjl_update(KitJlApp* h)
    {
        if (!h || !h->app)
            return;
        try
        {
            h->app->update();
        }
        catch (const std::exception& e)
        {
            setError("kitjl_update", e.what());
        }
        catch (...)
        {
            setError("kitjl_update", "unknown C++ exception");
        }
    }

    int kitjl_is_running(KitJlApp* h)
    {
        if (!h || !h->app)
            return 0;
        try
        {
            return h->app->isRunning() ? 1 : 0;
        }
        catch (...)
        {
            return 0;
        }
    }

    void kitjl_post_quit(KitJlApp* h, int code)
    {
        if (!h || !h->app)
            return;
        try
        {
            h->app->postQuit(code);
        }
        catch (...)
        {
        }
    }

    // Orderly-ish shutdown: postQuit + a BOUNDED update pump, then free the
    // handle.  We deliberately do NOT call the blocking IApp::shutdown() nor
    // releaseFrameworkAndDeregisterBuiltins(): headless Kit teardown can stall
    // (kit_server.py hits the same and resorts to os._exit), and carb cannot
    // cleanly restart within a process anyway (one app per process — hazard b).
    // The Julia process exits right after, which reclaims the GPU/handles.
    int kitjl_shutdown(KitJlApp* h)
    {
        if (!h)
            return 0;
        try
        {
            if (h->app && h->started)
            {
                h->app->postQuit(0);
                for (int i = 0; i < 60 && h->app->isRunning(); ++i)
                {
                    h->app->update();
                }
            }
        }
        catch (const std::exception& e)
        {
            setError("kitjl_shutdown", e.what());
        }
        catch (...)
        {
            setError("kitjl_shutdown", "unknown C++ exception");
        }
        delete h;
        return 0;
    }

    void kitjl_set_setting_bool(KitJlApp* h, const char* path, int value)
    {
        auto* s = settingsOf(h);
        if (!s)
            return;
        try
        {
            s->setBool(path, value != 0);
        }
        catch (...)
        {
        }
    }

    void kitjl_set_setting_int(KitJlApp* h, const char* path, long long value)
    {
        auto* s = settingsOf(h);
        if (!s)
            return;
        try
        {
            s->setInt64(path, static_cast<int64_t>(value));
        }
        catch (...)
        {
        }
    }

    void kitjl_set_setting_float(KitJlApp* h, const char* path, double value)
    {
        auto* s = settingsOf(h);
        if (!s)
            return;
        try
        {
            s->setFloat64(path, value);
        }
        catch (...)
        {
        }
    }

    void kitjl_set_setting_string(KitJlApp* h, const char* path, const char* value)
    {
        auto* s = settingsOf(h);
        if (!s)
            return;
        try
        {
            s->setString(path, value);
        }
        catch (...)
        {
        }
    }

    int kitjl_get_setting_bool(KitJlApp* h, const char* path)
    {
        auto* s = settingsOf(h);
        if (!s)
            return 0;
        try
        {
            return s->getAsBool(path) ? 1 : 0;
        }
        catch (...)
        {
            return 0;
        }
    }

    // Run a Python string in Kit's embedded interpreter (the scripting escape
    // hatch used for stage open / set_attr / capture / write_vdb, reusing
    // kit_server.py's proven handler bodies).  Returns 0 on success.
    int kitjl_exec_string(KitJlApp* h, const char* code)
    {
        if (!h || !h->scripting)
        {
            setError("kitjl_exec_string", "no scripting interface");
            return 1;
        }
        try
        {
            bool ok = h->scripting->executeString(code);
            if (!ok)
                setError("kitjl_exec_string", "executeString returned false");
            return ok ? 0 : 1;
        }
        catch (const std::exception& e)
        {
            setError("kitjl_exec_string", e.what());
            return 1;
        }
        catch (...)
        {
            setError("kitjl_exec_string", "unknown C++ exception");
            return 1;
        }
    }

} // extern "C"
