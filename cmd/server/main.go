// Command server is the FinRAG HTTP entrypoint. It reads configuration from the
// environment, wires the storage/embedding/RAG layers together, and serves the
// REST API with graceful shutdown.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ahmedfawzy21/finrag/internal/api"
	"github.com/ahmedfawzy21/finrag/internal/ingestion"
	"github.com/ahmedfawzy21/finrag/internal/retrieval"
	"github.com/ahmedfawzy21/finrag/internal/storage"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	if err := run(logger); err != nil {
		logger.Error("server exited with error", "error", err)
		os.Exit(1)
	}
}

func run(logger *slog.Logger) error {
	// Required configuration, all via environment variables — no secrets in code.
	databaseURL, err := requireEnv("DATABASE_URL")
	if err != nil {
		return err
	}
	voyageKey, err := requireEnv("VOYAGE_API_KEY")
	if err != nil {
		return err
	}
	anthropicKey, err := requireEnv("ANTHROPIC_API_KEY")
	if err != nil {
		return err
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	store, err := storage.NewStore(ctx, databaseURL)
	if err != nil {
		return fmt.Errorf("failed to initialize storage: %w", err)
	}
	defer store.Close()
	logger.Info("connected to database and applied schema")

	embedder := ingestion.NewEmbedder(voyageKey)
	rag := retrieval.NewRAGEngine(store, embedder, anthropicKey)
	server := api.NewServer(store, embedder, rag, logger)

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: server.Handler(),
		// Generous read/write timeouts: uploads can be large and RAG queries
		// wait on the embedding + Claude generation round-trips.
		ReadTimeout:       60 * time.Second,
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      300 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	// Serve in the background so main can wait for a shutdown signal.
	serveErr := make(chan error, 1)
	go func() {
		logger.Info("starting HTTP server", "addr", httpServer.Addr)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
			return
		}
		serveErr <- nil
	}()

	// Wait for SIGINT/SIGTERM or a fatal serve error.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-serveErr:
		return err
	case sig := <-stop:
		logger.Info("shutdown signal received", "signal", sig.String())
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("failed to shut down gracefully: %w", err)
	}
	logger.Info("server stopped cleanly")
	return nil
}

// requireEnv returns the value of the named environment variable or a clear
// error if it is unset.
func requireEnv(name string) (string, error) {
	v := os.Getenv(name)
	if v == "" {
		return "", fmt.Errorf("required environment variable %s is not set", name)
	}
	return v, nil
}
