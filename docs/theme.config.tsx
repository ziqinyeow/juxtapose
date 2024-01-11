import Logo from "./components/Logo";

export const REPO_LINK = "https://github.com/ziqinyeow/juxt";

// eslint-disable-next-line import/no-anonymous-default-export
export default {
  logo: <Logo />,
  docsRepositoryBase: `${REPO_LINK}/tree/main/docs/`,
  project: {
    link: "https://github.com/ziqinyeow/juxtapose",
  },
  head: (
    <>
      <link rel="apple-touch-icon" sizes="180x180" href={`/favicon.ico`} />
      <link rel="icon" type="image/png" sizes="32x32" href={`/favicon.ico`} />
      <link rel="icon" type="image/png" sizes="16x16" href={`/favicon.ico`} />
      <link rel="mask-icon" href={`/favicon.ico`} color="#000000" />
      <link rel="shortcut icon" href={`/favicon.ico`} />,
    </>
  ),
  // ... other theme options
};
