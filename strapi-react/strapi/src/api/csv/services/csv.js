'use strict';

/**
 * csv service
 */

const { createCoreService } = require('@strapi/strapi').factories;

module.exports = createCoreService('api::csv.csv');
