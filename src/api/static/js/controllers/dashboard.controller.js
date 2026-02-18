(function () {
    "use strict";

    angular
        .module("abaxDashboard")
        .controller("DashboardController", DashboardController);

    DashboardController.$inject = ["$interval", "ApiService"];

    function DashboardController($interval, ApiService) {
        var vm = this;

        vm.tasks = [
            { value: "classification", label: "Classification (UAH-DriveSet)" },
            { value: "regression", label: "Regression (EPA Fuel Economy)" }
        ];

        vm.task = "classification";
        vm.activeTab = "studio";

        vm.metadata = null;
        vm.features = [];
        vm.numericFeatures = [];
        vm.models = [];
        vm.modelDetails = [];
        vm.featureLookup = {};
        vm.modelLookup = {};

        vm.featureSelection = null;
        vm.featurePair = { first: null, second: null };
        vm.modelSelection = null;
        vm.customLearning = {
            selectedFeatures: [],
            modelSelection: null,
            cvFolds: 1,
            featureQuery: "",
            persistArtifact: false,
            artifactId: ""
        };
        vm.availableCvFolds = [1, 3, 5];
        vm.selectedFeatureInfo = null;
        vm.selectedModelInfo = null;
        vm.customSelectedFeatureInfo = [];
        vm.customJobId = null;
        vm.customJobStatus = null;
        vm.customLearningPollPromise = null;

        vm.loading = {
            metadata: false,
            feature: false,
            pair: false,
            corr: false,
            diagnostics: false,
            compare: false,
            custom: false
        };

        vm.errors = {
            metadata: null,
            feature: null,
            pair: null,
            corr: null,
            diagnostics: null,
            compare: null,
            custom: null
        };

        vm.results = {
            feature: null,
            pair: null,
            corr: null,
            diagnostics: null,
            compare: null,
            custom: null
        };

        vm.resultKeys = {
            compare: []
        };

        vm.onTaskChange = onTaskChange;
        vm.runFeatureAnalysis = runFeatureAnalysis;
        vm.runTwoFeatureAnalysis = runTwoFeatureAnalysis;
        vm.runCorrelation = runCorrelation;
        vm.runDiagnostics = runDiagnostics;
        vm.runModelComparison = runModelComparison;
        vm.onFeatureChange = onFeatureChange;
        vm.onModelChange = onModelChange;
        vm.onCustomFeatureChange = onCustomFeatureChange;
        vm.selectCustomFeatures = selectCustomFeatures;
        vm.customFeatureFilter = customFeatureFilter;
        vm.renderCell = renderCell;
        vm.runCustomLearning = runCustomLearning;

        init();

        function init() {
            loadTaskData();
        }

        function resetTaskScopedState() {
            vm.results.feature = null;
            vm.results.pair = null;
            vm.results.corr = null;
            vm.results.diagnostics = null;
            vm.results.compare = null;
            vm.results.custom = null;
            vm.resultKeys.compare = [];
            vm.selectedFeatureInfo = null;
            vm.selectedModelInfo = null;
            vm.customSelectedFeatureInfo = [];
            vm.featureLookup = {};
            vm.modelLookup = {};
            vm.customJobId = null;
            vm.customJobStatus = null;
            stopCustomJobPolling();

            vm.errors.feature = null;
            vm.errors.pair = null;
            vm.errors.corr = null;
            vm.errors.diagnostics = null;
            vm.errors.compare = null;
            vm.errors.custom = null;
        }

        function onTaskChange() {
            resetTaskScopedState();
            loadTaskData();
        }

        function loadTaskData() {
            vm.loading.metadata = true;
            vm.errors.metadata = null;

            ApiService.getMetadata(vm.task)
                .then(function (response) {
                    vm.metadata = response.data;
                    vm.features = response.data.features || [];
                    vm.numericFeatures = vm.features.filter(function (feature) {
                        return feature.is_numeric;
                    });
                    vm.models = response.data.models || [];
                    vm.modelDetails = response.data.model_details || [];
                    vm.featureLookup = buildLookup(vm.features, "name");
                    vm.modelLookup = buildLookup(vm.modelDetails, "name");

                    vm.featureSelection = vm.features.length ? vm.features[0].name : null;
                    vm.modelSelection = vm.models.length ? vm.models[0] : null;
                    onFeatureChange();
                    onModelChange();
                    vm.customLearning.modelSelection = vm.modelSelection;
                    vm.customLearning.cvFolds = 1;
                    vm.customLearning.featureQuery = "";
                    vm.customLearning.persistArtifact = false;
                    vm.customLearning.artifactId = "";
                    selectCustomFeatures(vm.numericFeatures.length ? "numeric" : "all");
                    onCustomFeatureChange();

                    if (vm.numericFeatures.length >= 2) {
                        vm.featurePair.first = vm.numericFeatures[0].name;
                        vm.featurePair.second = vm.numericFeatures[1].name;
                    } else if (vm.numericFeatures.length === 1) {
                        vm.featurePair.first = vm.numericFeatures[0].name;
                        vm.featurePair.second = vm.numericFeatures[0].name;
                    } else {
                        vm.featurePair.first = null;
                        vm.featurePair.second = null;
                    }
                })
                .catch(function (error) {
                    vm.errors.metadata = parseError(error, "Failed to load task metadata.");
                })
                .finally(function () {
                    vm.loading.metadata = false;
                });
        }

        function buildLookup(items, key) {
            var lookup = {};
            (items || []).forEach(function (item) {
                lookup[item[key]] = item;
            });
            return lookup;
        }

        function onFeatureChange() {
            vm.selectedFeatureInfo = vm.featureLookup[vm.featureSelection] || null;
        }

        function onModelChange() {
            vm.selectedModelInfo = vm.modelLookup[vm.modelSelection] || null;
        }

        function onCustomFeatureChange() {
            vm.customSelectedFeatureInfo = (vm.customLearning.selectedFeatures || [])
                .map(function (name) {
                    return vm.featureLookup[name] || null;
                })
                .filter(function (item) {
                    return item !== null;
                });
        }

        function selectCustomFeatures(mode) {
            if (mode === "clear") {
                vm.customLearning.selectedFeatures = [];
                onCustomFeatureChange();
                return;
            }

            var source = mode === "numeric" ? vm.numericFeatures : vm.features;
            vm.customLearning.selectedFeatures = (source || []).slice(0, 12).map(function (feature) {
                return feature.name;
            });
            onCustomFeatureChange();
        }

        function customFeatureFilter(feature) {
            if (!vm.customLearning.featureQuery) {
                return true;
            }
            var query = vm.customLearning.featureQuery.toLowerCase();
            var haystack = [feature.name, feature.description, feature.source_type, feature.source_summary]
                .join(" ")
                .toLowerCase();
            return haystack.indexOf(query) !== -1;
        }

        function stopCustomJobPolling() {
            if (vm.customLearningPollPromise) {
                $interval.cancel(vm.customLearningPollPromise);
                vm.customLearningPollPromise = null;
            }
        }

        function pollCustomJob() {
            stopCustomJobPolling();
            vm.customLearningPollPromise = $interval(function () {
                ApiService.getJobStatus(vm.customJobId)
                    .then(function (response) {
                        var payload = response.data || {};
                        vm.customJobStatus = payload.status || "running";

                        if (payload.status === "completed") {
                            vm.results.custom = payload.result || null;
                            if (vm.results.custom && vm.results.custom.selected_features) {
                                vm.customSelectedFeatureInfo = vm.results.custom.selected_features;
                            }
                            vm.loading.custom = false;
                            stopCustomJobPolling();
                        } else if (payload.status === "failed") {
                            vm.errors.custom = payload.error || "Custom learning job failed.";
                            vm.loading.custom = false;
                            stopCustomJobPolling();
                        }
                    })
                    .catch(function (error) {
                        vm.errors.custom = parseError(error, "Job polling failed.");
                        vm.loading.custom = false;
                        stopCustomJobPolling();
                    });
            }, 1000);
        }

        function runFeatureAnalysis() {
            if (!vm.featureSelection) {
                vm.errors.feature = "Select a feature first.";
                return;
            }

            vm.loading.feature = true;
            vm.errors.feature = null;
            vm.results.feature = null;

            ApiService.analyzeFeature({
                task: vm.task,
                feature_name: vm.featureSelection
            })
                .then(function (response) {
                    vm.results.feature = response.data;
                    vm.selectedFeatureInfo = response.data.feature_info || vm.selectedFeatureInfo;
                })
                .catch(function (error) {
                    vm.errors.feature = parseError(error, "Feature analysis failed.");
                })
                .finally(function () {
                    vm.loading.feature = false;
                });
        }

        function runTwoFeatureAnalysis() {
            if (!vm.featurePair.first || !vm.featurePair.second) {
                vm.errors.pair = "Select two numeric features.";
                return;
            }

            vm.loading.pair = true;
            vm.errors.pair = null;
            vm.results.pair = null;

            ApiService.analyzeTwoFeatures({
                task: vm.task,
                feature1: vm.featurePair.first,
                feature2: vm.featurePair.second
            })
                .then(function (response) {
                    vm.results.pair = response.data;
                })
                .catch(function (error) {
                    vm.errors.pair = parseError(error, "Two-feature analysis failed.");
                })
                .finally(function () {
                    vm.loading.pair = false;
                });
        }

        function runCorrelation() {
            vm.loading.corr = true;
            vm.errors.corr = null;
            vm.results.corr = null;

            ApiService.getCorrelationMatrix(vm.task)
                .then(function (response) {
                    vm.results.corr = response.data;
                })
                .catch(function (error) {
                    vm.errors.corr = parseError(error, "Correlation matrix loading failed.");
                })
                .finally(function () {
                    vm.loading.corr = false;
                });
        }

        function runDiagnostics() {
            if (!vm.modelSelection) {
                vm.errors.diagnostics = "Select a model first.";
                return;
            }

            vm.loading.diagnostics = true;
            vm.errors.diagnostics = null;
            vm.results.diagnostics = null;

            var request;
            if (vm.task === "classification") {
                request = ApiService.getConfusionMatrix({
                    task: vm.task,
                    model_name: vm.modelSelection
                });
            } else {
                request = ApiService.getRegressionDiagnostics({
                    task: vm.task,
                    model_name: vm.modelSelection
                });
            }

            request
                .then(function (response) {
                    var payload = response.data;
                    if (vm.task === "classification") {
                        payload.metrics = angular.extend({ accuracy: payload.accuracy }, payload.metrics);
                    }
                    vm.results.diagnostics = payload;
                })
                .catch(function (error) {
                    vm.errors.diagnostics = parseError(error, "Model diagnostics failed.");
                })
                .finally(function () {
                    vm.loading.diagnostics = false;
                });
        }

        function runModelComparison() {
            vm.loading.compare = true;
            vm.errors.compare = null;
            vm.results.compare = null;
            vm.resultKeys.compare = [];

            ApiService.compareModels(vm.task)
                .then(function (response) {
                    vm.results.compare = response.data;
                    if (response.data.results && response.data.results.length > 0) {
                        vm.resultKeys.compare = Object.keys(response.data.results[0]);
                    }
                })
                .catch(function (error) {
                    vm.errors.compare = parseError(error, "Model comparison failed.");
                })
                .finally(function () {
                    vm.loading.compare = false;
                });
        }

        function runCustomLearning() {
            if (!vm.customLearning.modelSelection) {
                vm.errors.custom = "Select a model for custom learning.";
                return;
            }
            var hasArtifact = !!(vm.customLearning.artifactId && vm.customLearning.artifactId.trim());
            if (!hasArtifact && (!vm.customLearning.selectedFeatures || vm.customLearning.selectedFeatures.length === 0)) {
                vm.errors.custom = "Select at least one feature.";
                return;
            }

            vm.loading.custom = true;
            vm.errors.custom = null;
            vm.results.custom = null;

            ApiService.startCustomLearningJob({
                task: vm.task,
                model_name: vm.customLearning.modelSelection,
                feature_names: vm.customLearning.selectedFeatures,
                cv_folds: vm.customLearning.cvFolds,
                persist_artifact: vm.customLearning.persistArtifact,
                artifact_id: vm.customLearning.artifactId || null
            })
                .then(function (response) {
                    vm.customJobId = response.data.job_id;
                    vm.customJobStatus = response.data.status;
                    pollCustomJob();
                })
                .catch(function (error) {
                    vm.errors.custom = parseError(error, "Custom learning run failed.");
                    vm.loading.custom = false;
                })
                ;
        }

        function parseError(error, fallback) {
            if (error && error.data) {
                if (typeof error.data.detail === "string") {
                    return error.data.detail;
                }
                if (Array.isArray(error.data.detail) && error.data.detail.length > 0) {
                    return error.data.detail[0].msg || fallback;
                }
            }
            return fallback;
        }

        function renderCell(value) {
            if (typeof value === "number" && isFinite(value)) {
                return value.toFixed(4);
            }
            return value;
        }
    }
})();
