# Arrival Classifier UI

A web-based UI for visualizing and classifying query arrival patterns.

## Features

- **File Browser**: Browse and select parquet files from the data directory
- **Time Series Visualization**: Interactive scatter plot showing query arrivals over time
- **Arrival Classification**: Classify templates as windowed, normal, or ad-hoc patterns
- **Summary Statistics**: View classification breakdown by templates and queries

## Quick Start

1. Start the development servers:
   ```bash
   ./start-dev.sh
   ```

2. Open your browser to http://localhost:5174

3. To stop:
   ```bash
   ./stop-dev.sh
   ```

## File Format

The UI reads parquet files from the `data` directory. Files should follow the schema
defined in `src/autoslo/utils/data_schemas.yml`:

### Required columns:
- `rel_start_time_s` (or `start_time_s`): Relative arrival time in seconds
- `query_template` (or `template_id`): Query template identifier

### Optional columns:
- `query_id`: Unique query identifier (auto-generated if missing)
- `query_num_within_template`: Query number within template

## API Endpoints

The UI communicates with the backend API at `http://localhost:1998/api/classifier/`:

- `GET /api/classifier/files` - List available parquet files in the data directory
- `GET /api/classifier/load?path=...` - Load a parquet file for visualization
- `POST /api/classifier/classify-file?path=...` - Classify arrivals from a file
- `POST /api/classifier/classify-arrivals` - Classify already-loaded arrivals

## Development

### Frontend

Built with:
- React 19
- Vite
- Chart.js with react-chartjs-2

### Backend

The classifier API is part of the main AutoSLO API (`src/autoslo/api`).

### Project Structure

```
uis/classifier/
├── src/
│   ├── App.jsx       # Main application component
│   ├── App.css       # Styles
│   ├── main.jsx      # Entry point
│   └── index.css     # Global styles
├── index.html
├── package.json
├── vite.config.js
├── start-dev.sh
└── stop-dev.sh
```

## Classification Types

- **Windowed**: Templates with regular, periodic arrival patterns (e.g., hourly batches)
- **Normal**: Templates with stable, consistent arrival patterns across time
- **Ad-hoc**: Templates with irregular, infrequent arrivals
