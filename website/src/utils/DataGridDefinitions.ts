interface ValueFormatterParam {
  value: number;
}

export const BenchmarkRegions = [
  {
    region: "iad",
    label: "US East (Virginia)",
    dateAdded: "2024-04-13",
  },
  {
    region: "cdg",
    label: "Europe (Paris)",
    dateAdded: "2024-04-13",
  },
  {
    region: "sea",
    label: "US West (Seattle)",
    dateAdded: "2024-04-13",
  },
];

export const ModelDefinition = {
  title: "模型名称",
  definition: "具体的模型型号。",
};

export const ProviderDefinition = {
  title: "API提供商",
  definition: "模型托管云服务商。",
};

export const TTFTDefinition = {
  title: "TTFT",
  definition:
    "首次生成 token 时间。这指的是模型处理进来的请求并开始输出文本的速度，直接关系到用户界面何时开始更新。数值越低，意味着延迟越低，性能越快。",
  bestPerformance: 0.2,
  worstPerformance: 0.5,
};

export const TPSDefinition = {
  title: "TPS",
  definition:
    "每秒生成 token 数。这是指模型生成文本的速度，控制着完整的响应在用户界面上显示的速度。数值越高，意味着吞吐量更大，性能更快。",
  bestPerformance: 100,
  worstPerformance: 30,
};

export const TotalTimeDefinition = {
  title: "Total",
  definition:
    "从请求开始到响应完成的总时间，即最后一个 token 生成的时间。总时间 = 首次生成 token 时间 + 每秒生成 token 数 * token 总数。数值越低，意味着延迟越低，性能越好。",
  bestPerformance: 0.4,
  worstPerformance: 1.0,
};

export const ContextDefinition = {
  title: "Context",
  definition:
    "模型支持的上下文长度",
};

export const InputPriceDefinition = {
  title: "输入价格",
  definition:
    "输入价格 元/百万tokens",
};

export const OutputPriceDefinition = {
  title: "输出价格",
  definition:
    "输出价格 元/百万tokens",
};

// Set-up all of our column definitions that will be used in the Data Grid
const headerClass = "font-bold";

function milliSecParser(text: string) {
  return parseFloat(text) / 1000;
}
function milliSecFormatter(value: number) {
  return (value * 1000).toString();
}

// Model column
const columnModel = {
  field: "model",
  headerName: ModelDefinition.title,
  headerTooltip: ModelDefinition.definition,
  headerClass: headerClass,
  //TODO: Make this ~200 on mobile screen size by default
  minWidth: 120,
  // tooltipField: "output"
  filterParams: {
    filterOptions: ["contains"],
    maxNumConditions: 4,
  },
};

const columnProvider = {
  field: "provider",
  headerName: ProviderDefinition.title,
  headerTooltip: ProviderDefinition.definition,
  headerClass: headerClass,
  minWidth: 100,
  maxWidth: 150,
  filterParams: {
    filterOptions: ["contains"],
    maxNumConditions: 4,
  },
};

// TTFT column
const columnTTFT = {
  field: "ttft",
  headerName: TTFTDefinition.title,
  headerTooltip: TTFTDefinition.definition,
  headerClass: headerClass,
  minWidth: 0,
  maxWidth: 120,
  valueFormatter: (p: ValueFormatterParam) =>
    p.value < 1.0 ? p.value.toFixed(3) * 1000 + "ms" : p.value.toFixed(2) + "s",
  filterParams: {
    filterOptions: ["lessThanOrEqual"],
    maxNumConditions: 1,
    numberParser: milliSecParser,
    numberFormatter: milliSecFormatter,
  },
};

// TPS column
const columnTPS = {
  field: "tps",
  headerName: TPSDefinition.title,
  headerTooltip: TPSDefinition.definition,
  headerClass: headerClass,
  minWidth: 0,
  maxWidth: 120,
  valueFormatter: (p: ValueFormatterParam) => p.value.toFixed(2),
  filterParams: {
    filterOptions: ["greaterThanOrEqual"],
    maxNumConditions: 1,
  },
};

// Total Time column
const columnTotalTime = {
  field: "total_time",
  headerName: TotalTimeDefinition.title,
  headerTooltip: TotalTimeDefinition.definition,
  headerClass: headerClass,
  minWidth: 80,
  maxWidth: 120,
  // minWidth: 100,
  // maxWidth: 100,
  wrapHeaderText: true,
  // valueFormatter: (p: ValueFormatterParam) => p.value.toFixed(2) + "s",
  valueFormatter: (p: ValueFormatterParam) =>
    p.value < 1.0 ? p.value.toFixed(3) * 1000 + "ms" : p.value.toFixed(2) + "s",
  sort: "asc",
  filterParams: {
    filterOptions: ["lessThanOrEqual"],
    maxNumConditions: 1,
    numberParser: milliSecParser,
    numberFormatter: milliSecFormatter,
  },
};

// Context column
const columnContext = {
  field: "context",
  headerName: ContextDefinition.title,
  headerTooltip: ContextDefinition.definition,
  headerClass: headerClass,
  minWidth: 0,
  maxWidth:120,
  filterParams: {
    filterParams: {
      filterOptions: ["greaterThanOrEqual"],
      maxNumConditions: 1,
    },
  },
};

// InputPrice column
const columnInputPrice = {
  field: "input_price",
  headerName: InputPriceDefinition.title,
  headerTooltip: InputPriceDefinition.definition,
  headerClass: headerClass,
  minWidth: 0,
  maxWidth:120,
  filterParams: {
    filterParams: {
      filterOptions: ["greaterThanOrEqual"],
      maxNumConditions: 1,
    },
  },
};

// OutputPrice column
const columnOutputPrice = {
  field: "output_price",
  headerName: OutputPriceDefinition.title,
  headerTooltip: OutputPriceDefinition.definition,
  headerClass: headerClass,
  minWidth: 0,
  maxWidth:120,
  filterParams: {
    filterParams: {
      filterOptions: ["greaterThanOrEqual"],
      maxNumConditions: 1,
    },
  },
};

export const gridOptionsBase = {
  // alwaysShowVerticalScroll: true,
  autoSizeStrategy: { type: "fitGridWidth" },
  enableCellTextSelection: true,
  suppressCellFocus: true,
  suppressRowHoverHighlight: true,
  defaultColDef: {
    suppressMovable: true,
    filter: true,
    floatingFilter: true,
    suppressHeaderMenuButton: true,
    // minWidth: 80,
  },
  domLayout: "autoHeight",
  rowData: [],
  // Columns to be displayed (Should match rowData properties)...omit columnRegion, columnTTR, columnNumTokens
  columnDefs: [
    columnProvider,
    columnModel,
    columnTTFT,
    columnTPS,
    columnTotalTime,
    columnContext,
    columnInputPrice,
    columnOutputPrice
  ],
};
