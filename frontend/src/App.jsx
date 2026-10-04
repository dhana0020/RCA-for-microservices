import { useState } from "react";
import "./App.css";

const SERVICES = [
  "checkoutservice",
  "currencyservice",
  "emailservice",
  "productcatalogservice",
  "recommendationservice",
];

const createDefaultFeatures = () => ({
  cpu_change: 0,
  cpu_ratio: 1,

  mem_change: 0,
  mem_ratio: 1,

  diskio_change: 0,
  diskio_ratio: 1,

  socket_change: 0,
  socket_ratio: 1,

  workload_change: 0,
  workload_ratio: 1,

  error_change: 0,
  error_ratio: 1,

  latency_50_change: 0,
  latency_50_ratio: 1,

  latency_90_change: 0,
  latency_90_ratio: 1,

  log_count_change: 0,
  log_count_ratio: 1,
  log_error_count: 0,
  log_timeout_count: 0,
  log_connection_count: 0,

  trace_count_change: 0,
  trace_count_ratio: 1,
  trace_duration_change: 0,
  trace_duration_ratio: 1,
  trace_failure_count: 0,
});

function App() {
  const [selectedService, setSelectedService] =
    useState("checkoutservice");

  const [serviceFeatures, setServiceFeatures] = useState(() => {
    const data = {};

    SERVICES.forEach((service) => {
      data[service] = createDefaultFeatures();
    });

    return data;
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const currentFeatures = serviceFeatures[selectedService];

  const handleChange = (e) => {
    setServiceFeatures({
      ...serviceFeatures,
      [selectedService]: {
        ...serviceFeatures[selectedService],
        [e.target.name]: Number(e.target.value),
      },
    });
  };

  const analyze = async () => {
    setLoading(true);
    setResult(null);

    try {
      const services = SERVICES.map((serviceName) => ({
        service: serviceName,
        ...serviceFeatures[serviceName],
      }));

      console.log("Sending services:", services);

      const response = await fetch(
        "http://localhost:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            application: "RE2-OB",
            services: services,
          }),
        }
      );

      if (!response.ok) {
        const errorData = await response.json();

        console.error("API Error:", errorData);

        throw new Error("API request failed");
      }

      const data = await response.json();

      console.log("RCA Result:", data);

      setResult(data);
    } catch (error) {
      console.error("Error:", error);

      alert(
        "Could not connect to FastAPI. Check that Docker is running."
      );
    }

    setLoading(false);
  };

  return (
    <div className="app">

      <header>
        <h1>AI Root Cause Analysis</h1>

        <p>
          Microservices Incident Diagnosis Dashboard
        </p>
      </header>

      <main>

        {/* SERVICE SELECTION */}

        <section className="card">

          <h2>Microservice Telemetry</h2>

          <p>
            Enter telemetry features for each service.
            The model will compare all services and identify
            the most likely root cause.
          </p>

          <label>Select Service</label>

          <select
            value={selectedService}
            onChange={(e) =>
              setSelectedService(e.target.value)
            }
          >
            {SERVICES.map((service) => (
              <option key={service} value={service}>
                {service}
              </option>
            ))}
          </select>

          <h3>
            Features for: {selectedService}
          </h3>

          <div className="grid">

            {Object.keys(currentFeatures).map((key) => (
              <div key={key}>

                <label>{key}</label>

                <input
                  type="number"
                  step="any"
                  name={key}
                  value={currentFeatures[key]}
                  onChange={handleChange}
                />

              </div>
            ))}

          </div>

          <button
            onClick={analyze}
            disabled={loading}
          >
            {loading
              ? "Analyzing..."
              : "Analyze Root Cause"}
          </button>

        </section>


        {/* RESULT */}

        {result && (

          <section className="card result">

            <h2>Analysis Result</h2>


            <div className="result-box">

              <h3>Root Cause Service</h3>

              <strong>
                {result.root_cause_service}
              </strong>

            </div>


            <div className="result-box">

              <h3>Fault Type</h3>

              <strong>
                {result.fault_type}
              </strong>

            </div>


            <div className="result-box">

              <h3>Root Cause Confidence</h3>

              <strong>
                {(
                  result.root_cause_confidence * 100
                ).toFixed(2)}
                %
              </strong>

            </div>


            <div className="result-box">

              <h3>Fault Confidence</h3>

              <strong>
                {(
                  result.fault_confidence * 100
                ).toFixed(2)}
                %
              </strong>

            </div>


            {/* SERVICE RANKING */}

            {result.service_ranking && (

              <div className="ranking">

                <h2>Service Ranking</h2>

                {result.service_ranking.map(
                  (item, index) => {

                    const serviceName =
                      item.service ||
                      item.service_name;

                    const probability =
                      item.probability ??
                      item.confidence ??
                      item.score ??
                      0;

                    return (
                      <div
                        className="ranking-row"
                        key={serviceName}
                      >

                        <span className="rank">
                          #{index + 1}
                        </span>

                        <span className="service-name">
                          {serviceName}
                        </span>

                        <span className="score">
                          {(probability * 100).toFixed(2)}%
                        </span>

                      </div>
                    );
                  }
                )}

              </div>

            )}

          </section>

        )}

      </main>

    </div>
  );
}

export default App;