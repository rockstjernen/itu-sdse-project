package main

import (
	"context"
	"dagger/itu-sdse-project/internal/dagger"
)

type ItuSdseProject struct{}

// Train builds the container and runs the training pipeline, returning the model file
func (m *ItuSdseProject) Train(ctx context.Context, source *dagger.Directory) *dagger.File {
	// Build container from source directory using our Dockerfile
	container := dag.Container().
		Build(source, dagger.ContainerBuildOpts{
			Dockerfile: "train.dockerfile",
		})

	// Run the container (executes the CMD from Dockerfile)
	trained := container.WithExec([]string{"python", "-m", "itu_sdse_project.pipeline"})

	// Return the model file
	return trained.File("/app/models/model.pkl")
}
