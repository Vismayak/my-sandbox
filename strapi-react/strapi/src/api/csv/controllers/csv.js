'use strict';

/**
 * csv controller
 */

const { createCoreController } = require('@strapi/strapi').factories;

module.exports = createCoreController('api::csv.csv');
