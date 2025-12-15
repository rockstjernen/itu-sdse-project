package main

import (
	"context"
	"dagger/itu-sdse-project/internal/dagger"
)

type ItuSdseProject struct{}

// Train builds the container and runs the training pipeline, returning the model file
func (m *ItuSdseProject) Train(ctx context.Context, source *dagger.Directory, githubToken *dagger.Secret) *dagger.File {
	// Build container from source directory using our Dockerfile
	container := dag.Container().
		Build(source, dagger.ContainerBuildOpts{
			Dockerfile: "train.dockerfile",
		}).
		WithSecretVariable("GITHUB_TOKEN", githubToken).
		WithExec([]string{"sh", "-c", "git config --global url.\"https://${GITHUB_TOKEN}@github.com/\".insteadOf \"https://github.com/\""}).
		// Initialize git repo (required for DVC)
		WithExec([]string{"git", "init"}).
		WithExec([]string{"git", "config", "user.email", "ci@example.com"}).
		WithExec([]string{"git", "config", "user.name", "CI"}).
		// Pull data with DVC
		WithExec([]string{"dvc", "pull", "-v"})

	// Run the pipeline
	trained := container.WithExec([]string{"python", "-m", "itu_sdse_project.pipeline"})

	// Return the model file
	return trained.File("/app/models/model.pkl")
}
