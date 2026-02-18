(function () {
    "use strict";

    angular
        .module("abaxDashboard")
        .service("ApiService", ApiService);

    ApiService.$inject = ["$http"];

    function ApiService($http) {
        this.getMetadata = function (task) {
            return $http.get("/api/metadata", { params: { task: task } });
        };

        this.getFeatures = function (task) {
            return $http.get("/api/features", { params: { task: task } });
        };

        this.getModels = function (task) {
            return $http.get("/api/models", { params: { task: task } });
        };

        this.analyzeFeature = function (payload) {
            return $http.post("/api/feature", payload);
        };

        this.analyzeTwoFeatures = function (payload) {
            return $http.post("/api/two-features", payload);
        };

        this.getCorrelationMatrix = function (task) {
            return $http.get("/api/correlation-matrix", { params: { task: task } });
        };

        this.getConfusionMatrix = function (payload) {
            return $http.post("/api/model/confusion-matrix", payload);
        };

        this.getRegressionDiagnostics = function (payload) {
            return $http.post("/api/model/regression-diagnostics", payload);
        };

        this.runCustomLearning = function (payload) {
            return $http.post("/api/model/custom-learning", payload);
        };

        this.compareModels = function (task) {
            return $http.get("/api/model/compare", { params: { task: task } });
        };
    }
})();
