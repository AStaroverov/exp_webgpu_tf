export type TWGSLModule = TWGSLPart | TWGSLFunc;

export type TWGSLPart = {
    deps: TWGSLModule[];
    body: string;
};

export type TWGSLFunc = {
    deps: TWGSLModule[];
    name: string;
    body: string;
}