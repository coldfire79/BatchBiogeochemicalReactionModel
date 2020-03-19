/*
A KBase module: BatchBiogeochemicalReactionModel
*/

module BatchBiogeochemicalReactionModel {
    typedef structure {
        string report_name;
        string report_ref;
    } ReportResults;

    /*
        This example function accepts any number of parameters and returns results in a KBaseReport
    */
    funcdef run_BatchBiogeochemicalReactionModel(mapping<string,UnspecifiedObject> params) returns (ReportResults output) authentication required;

};
