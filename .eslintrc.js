module.exports = {
    parser: '@typescript-eslint/parser',
    parserOptions: {
        ecmaVersion: 2018,
        sourceType: 'module',
        ecmaFeatures: {
            jsx: true, // Allows for the parsing of JSX
        },
    },
    plugins: ['unused-imports', 'simple-import-sort', 'prettier'],
    settings: {
        react: {
            version: 'detect', // Tells eslint-plugin-react to automatically detect the version of React to use
        },
    },
    extends: [
        'plugin:import/recommended',
        'plugin:import/typescript',
        'plugin:react/recommended', // Uses the recommended rules from @eslint-plugin-react
        'plugin:@typescript-eslint/recommended', // Uses the recommended rules from the @typescript-eslint/eslint-plugin
        'prettier',
    ],
    rules: {
        'prettier/prettier': ['error', require('./.prettierrc.js')],

        'no-unused-vars': 'off',

        'import/no-unresolved': 'off',
        'import/no-default-export': 'error',
        'unused-imports/no-unused-imports': 'error',
        'simple-import-sort/imports': 'error',

        '@typescript-eslint/no-unused-vars': 'error',
        '@typescript-eslint/no-non-null-assertion': 'off',
        '@typescript-eslint/ban-ts-comment': 'off',
        '@typescript-eslint/no-var-requires': 'off',
        '@typescript-eslint/ban-types': 'off',

        'react/display-name': 'off',
        'react/prop-types': 'off',

        'no-debugger': 'error',
    },
    ignorePatterns: ['**/generated/*.ts'],
};
