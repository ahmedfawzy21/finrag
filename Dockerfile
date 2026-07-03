# --- build stage ---
# Note: the module's go directive is go 1.26.4, so the builder must be >= 1.26
# (the spec suggested golang:1.23, which cannot build this module).
FROM golang:1.26 AS build

WORKDIR /src

# Cache dependencies first.
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Static, CGO-free binary so it runs on a minimal runtime image.
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o /out/finrag ./cmd/server

# --- runtime stage ---
# Alpine (not distroless/static) because the PDF extractor shells out to
# `pdftotext` from poppler-utils, which must be present in the runtime image.
FROM alpine:3.20

RUN apk add --no-cache poppler-utils ca-certificates

COPY --from=build /out/finrag /usr/local/bin/finrag

EXPOSE 8080

ENTRYPOINT ["finrag"]
